#!/usr/bin/env python3
"""
Print XLeVR -> ee.delta action at TCP control rate (default 30Hz).

VR browser sends ~72-90Hz (A-Frame tick), but this script only samples/sends
at --control-fps, matching lerobot-record / TCP rate.

Usage:
    python examples/xlevr_to_a10/test_action_print.py
    python examples/xlevr_to_a10/test_action_print.py --control-fps 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from lerobot.teleoperators.xlevr.config_xlevr import XLeVRTeleopConfig
from lerobot.teleoperators.xlevr.factory import make_xlevr_a10_processors
from lerobot.teleoperators.xlevr.teleop_xlevr import XLeVRTeleop
from lerobot.utils.robot_utils import precise_sleep

XLEVR_PATH = "/home/robot/VLA/XLeRobot/XLeVR"
FAKE_OBS = {f"joint_{i}.pos": 0.0 for i in range(1, 7)} | {"gripper.pos": 0.0}
TCP_KEYS = (
    "ee.enabled",
    "ee.delta_x",
    "ee.delta_y",
    "ee.delta_z",
    "ee.delta_rx",
    "ee.delta_ry",
    "ee.delta_rz",
    "gripper.pos",
    "vr.button_squeeze",
    "vr.thumbstick_x",
)


# SET_EE_DELTA is EE local: +X up, +Y right, +Z forward
DIM_LABELS_POS = {
    "dx": ("向上", "向下"),
    "dy": ("向右", "向左"),
    "dz": ("向前", "向后"),
}
DIM_LABELS_ROT = {
    "rx": ("绕X逆时针", "绕X顺时针"),
    "ry": ("绕Y逆时针", "绕Y顺时针"),
    "rz": ("绕Z逆时针", "绕Z顺时针"),
}
RPY_HINT = "EE局部: 手柄上→+X 右→+Y 前→+Z | yaw/pitch/roll 跟手柄转"


def _describe_axis_group(
    vals: list[float],
    dims: tuple[str, ...],
    labels: dict[str, tuple[str, str]],
    threshold: float,
    unit: str,
    group_name: str,
) -> str:
    if not vals:
        return f"{group_name}: 无数据"

    max_idx = max(range(len(vals)), key=lambda i: abs(vals[i]))
    dim = dims[max_idx]
    value = vals[max_idx]
    abs_val = abs(value)

    if abs_val < threshold:
        return f"{group_name}: 静止 (max|.|={abs_val:.6f}{unit} < {threshold}{unit})"

    label = labels[dim][0 if value > 0 else 1]
    others = [
        f"{dims[i]}={vals[i]:+.4f}"
        for i in range(len(vals))
        if i != max_idx and abs(vals[i]) >= threshold
    ]
    secondary = f" | 次分量: {', '.join(others)}" if others else ""
    return (
        f"{group_name}: {label} ({dim}={value:+.6f}{unit}, |max|={abs_val:.6f}{unit})"
        f"{secondary}"
    )


def describe_dominant_motion(
    actions: list[float],
    *,
    enabled: bool,
    pos_threshold: float = 0.0005,
    angle_threshold: float = 0.05,
) -> list[str]:
    """Describe translation and rotation separately (different units)."""
    if len(actions) < 6:
        return ["数据不足"]

    if not enabled:
        return ["手臂未启用 (按住 squeeze)"]

    pos_vals = [float(v) for v in actions[:3]]
    rot_vals = [float(v) for v in actions[3:6]]
    return [
        _describe_axis_group(
            pos_vals,
            ("dx", "dy", "dz"),
            DIM_LABELS_POS,
            pos_threshold,
            "m",
            "平移",
        ),
        _describe_axis_group(
            rot_vals,
            ("rx", "ry", "rz"),
            DIM_LABELS_ROT,
            angle_threshold,
            "rad",
            "姿态(rotvec)",
        ),
    ]


def describe_gripper(gripper: float, threshold: float = 0.05) -> str | None:
    if abs(gripper) < threshold:
        return None
    if gripper > 0:
        return f"夹爪摇杆: 正方向 (gripper={gripper:+.3f})"
    return f"夹爪摇杆: 负方向 (gripper={gripper:+.3f})"


def action_to_tcp_payload(action: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(action.get("ee.enabled", False))
    if enabled:
        arm = [
            float(action.get("ee.delta_x", 0.0)),
            float(action.get("ee.delta_y", 0.0)),
            float(action.get("ee.delta_z", 0.0)),
            float(action.get("ee.delta_rx", 0.0)),
            float(action.get("ee.delta_ry", 0.0)),
            float(action.get("ee.delta_rz", 0.0)),
        ]
    else:
        arm = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return {"actions": arm + [float(action.get("gripper.pos", 0.0))]}


def is_meaningful_tcp(payload: dict[str, Any], thumbstick_x: float = 0.0) -> bool:
    actions = payload.get("actions", [])
    if len(actions) >= 7 and any(abs(float(v)) > 1e-9 for v in actions[:6]):
        return True
    if abs(thumbstick_x) > 0.05:
        return True
    if len(actions) >= 7 and abs(float(actions[6])) > 0.05:
        return True
    return False


def print_status(status: dict[str, Any], control_fps: int) -> None:
    print(
        f"[status] phase={status.get('phase')} | ws={status.get('ws_clients')} | "
        f"vr_rx={status.get('goals_received')} | tcp_fps={control_fps} | has_right={status.get('has_right_goal')}",
        flush=True,
    )
    print(f"         {status.get('hint')}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Print XLeVR TCP action at control rate")
    parser.add_argument("--xlevr-path", default=XLEVR_PATH)
    parser.add_argument("--arm", default="right", choices=["left", "right"])
    parser.add_argument("--control-fps", type=int, default=30, help="与 TCP/record 一致的控制频率")
    parser.add_argument("--status-interval", type=float, default=3.0)
    parser.add_argument("--print-idle", action="store_true", help="无运动时 also 打印零 delta")
    parser.add_argument("--pos-threshold", type=float, default=0.0005, help="平移方向判定阈值 (m)")
    parser.add_argument("--angle-threshold", type=float, default=0.05, help="旋转方向判定阈值 (deg)")
    return parser.parse_args()


def main():
    args = parse_args()

    teleop_config = XLeVRTeleopConfig(
        xlevr_path=args.xlevr_path,
        arm=args.arm,
        control_fps=args.control_fps,
    )
    teleop = XLeVRTeleop(teleop_config)
    teleop_action_processor, robot_action_processor, _ = make_xlevr_a10_processors(teleop_config)

    print("=" * 60, flush=True)
    print("XLeVR TCP action 测试", flush=True)
    print(f"  VR 浏览器发送: ~72-90 Hz (A-Frame tick)", flush=True)
    print(f"  本脚本/TCP 控制: {args.control_fps} Hz", flush=True)
    print("  按住 squeeze 才控制机械臂 | 右手摇杆 x 直接作为夹爪指令", flush=True)
    print("  机器人坐标: EE局部 +X上 +Y右 +Z前 | 平移/姿态分开判定方向", flush=True)
    print(f"  {RPY_HINT}", flush=True)
    print("=" * 60, flush=True)

    try:
        teleop.connect()
    except Exception as exc:
        print(f"启动失败: {exc}", flush=True)
        sys.exit(1)

    time.sleep(1.0)
    print_status(teleop.get_status(), args.control_fps)
    print("Ctrl+C 停止\n", flush=True)

    tick = 0
    last_status_t = 0.0
    control_period = 1.0 / args.control_fps

    try:
        while True:
            loop_start = time.perf_counter()

            if loop_start - last_status_t >= args.status_interval:
                print_status(teleop.get_status(), args.control_fps)
                last_status_t = loop_start

            raw = teleop.get_action()
            processed = teleop_action_processor((raw, FAKE_OBS))
            robot_action = robot_action_processor((processed, FAKE_OBS))
            tcp_payload = action_to_tcp_payload(robot_action)

            if args.print_idle or is_meaningful_tcp(
                tcp_payload, float(robot_action.get("vr.thumbstick_x", 0.0))
            ):
                actions = tcp_payload.get("actions", [])
                enabled = bool(robot_action.get("ee.enabled", False))
                gripper = float(actions[6]) if len(actions) >= 7 else 0.0

                print(f"\n--- tcp_tick {tick} @ {args.control_fps}Hz ---", flush=True)
                for line in describe_dominant_motion(
                    actions,
                    enabled=enabled,
                    pos_threshold=args.pos_threshold,
                    angle_threshold=args.angle_threshold,
                ):
                    print(f"  >> {line}", flush=True)
                grip_hint = describe_gripper(gripper)
                if grip_hint:
                    print(f"  >> {grip_hint}", flush=True)
                for key in TCP_KEYS:
                    if key in robot_action:
                        print(f"  {key}: {robot_action[key]}", flush=True)
                print(f"  SET_EE_DELTA: {json.dumps(tcp_payload, ensure_ascii=False)}", flush=True)

            tick += 1
            precise_sleep(max(0.0, control_period - (time.perf_counter() - loop_start)))
    except KeyboardInterrupt:
        print("\n已停止", flush=True)
    finally:
        teleop.disconnect()


if __name__ == "__main__":
    main()
