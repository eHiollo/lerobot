#!/usr/bin/env python3
"""
Teleoperate A10 with XLeVR — body-frame quaternion deltas -> SET_EE_DELTA (7D rotvec).

Robot frame: EE local +X up, +Y right, +Z forward (independent of base mount).

Usage:
    cd /home/allen/Allen/lerobot

    python examples/xlevr_to_a10/teleoperate.py --robot-host 192.168.1.12 --robot-port 8080
    python examples/xlevr_to_a10/teleoperate.py --robot-host 192.168.1.12 --no-reconnect
    python examples/xlevr_to_a10/teleoperate.py --vr-only
"""

import argparse
import logging
import time

from lerobot.robots.a10_follower.a10_follower import A10Follower
from lerobot.robots.a10_follower.config_a10_follower import A10FollowerConfig
from lerobot.teleoperators.xlevr.config_xlevr import XLeVRTeleopConfig
from lerobot.teleoperators.xlevr.diagnostics import (
    XLeVRDiagnosticsWriter,
    get_xlevr_diagnostics,
)
from lerobot.teleoperators.xlevr.factory import make_xlevr_a10_processors
from lerobot.teleoperators.xlevr.teleop_xlevr import XLeVRTeleop
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.robot_utils import precise_sleep

ROBOT_LINK_ERRORS = (
    ConnectionError,
    DeviceNotConnectedError,
    OSError,
    BrokenPipeError,
    TimeoutError,
)

FPS = 30
XLEVR_PATH = "/home/robot/VLA/XLeRobot/XLeVR"
DEFAULT_JOINTS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper")


def fake_observation() -> dict:
    return {f"{name}.pos": 0.0 for name in DEFAULT_JOINTS}


def parse_args():
    parser = argparse.ArgumentParser(description="XLeVR teleoperation for A10")
    parser.add_argument("--vr-only", action="store_true", help="只测 VR：不连机器人、不发 SET_EE_DELTA")
    parser.add_argument("--robot-host", default="192.168.1.6")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--robot-timeout-ms", type=int, default=300000, help="机器人 TCP 连接超时")
    parser.add_argument("--xlevr-path", default=XLEVR_PATH)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument(
        "--diagnostics-jsonl",
        help="逐帧保存原始 VR、过滤原因、滤波/限幅值和最终命令",
    )
    parser.add_argument(
        "--with-cameras",
        action="store_true",
        help="启用 config 里的 OpenCV 相机（默认不启，仅 TCP 发 actions）",
    )
    parser.add_argument("--print-every", type=int, default=30, help="vr-only 模式下每 N 帧打印一次 delta")
    parser.add_argument(
        "--reconnect",
        action="store_true",
        default=None,
        help="机器人 TCP 断线后自动重连（默认：非 --vr-only 时开启）",
    )
    parser.add_argument(
        "--no-reconnect",
        action="store_true",
        help="断线后不重连，进程退出",
    )
    parser.add_argument(
        "--reconnect-interval",
        type=float,
        default=2.0,
        metavar="SEC",
        help="断线后每隔多少秒尝试重连机器人",
    )
    args = parser.parse_args()
    if args.no_reconnect:
        args.reconnect = False
    elif args.reconnect is None:
        args.reconnect = not args.vr_only
    return args


def _safe_robot_disconnect(robot: A10Follower) -> None:
    try:
        if robot.is_connected:
            robot.disconnect()
    except Exception:
        pass


def try_robot_connect(robot: A10Follower, host: str, port: int) -> bool:
    _safe_robot_disconnect(robot)
    try:
        robot.connect()
        return True
    except ConnectionError as exc:
        logging.warning("机器人连接失败 %s:%s — %s", host, port, exc)
        return False


def wait_for_robot(
    robot: A10Follower,
    host: str,
    port: int,
    interval_s: float,
    *,
    first_attempt: bool = False,
) -> bool:
    attempt = 0
    while True:
        if first_attempt and attempt == 0:
            print(f"正在连接机器人 {host}:{port} ...")
        elif attempt > 0:
            print(f"等待机器人 {host}:{port}，{interval_s:.1f}s 后重试 ...")
        if try_robot_connect(robot, host, port):
            print(f"已连接机器人 @ {host}:{port}")
            return True
        attempt += 1
        try:
            time.sleep(interval_s)
        except KeyboardInterrupt:
            return False


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    teleop_config = XLeVRTeleopConfig(
        xlevr_path=args.xlevr_path,
        arm="right",
        control_fps=args.fps,
    )
    teleop = XLeVRTeleop(teleop_config)
    teleop_action_processor, robot_action_processor, _ = make_xlevr_a10_processors(teleop_config)
    diagnostics_writer = (
        XLeVRDiagnosticsWriter(args.diagnostics_jsonl)
        if args.diagnostics_jsonl
        else None
    )

    robot = None
    if not args.vr_only:
        robot_cfg_kwargs: dict = dict(
            host=args.robot_host,
            port=args.robot_port,
            timeout_ms=args.robot_timeout_ms,
            use_ee_delta=True,
        )
        if not args.with_cameras:
            robot_cfg_kwargs["cameras"] = {}

        robot = A10Follower(A10FollowerConfig(**robot_cfg_kwargs))

    print("正在启动 XLeVR...")
    teleop.connect()

    if args.vr_only:
        print("VR-only 模式：只打印 ee.delta，不连接机器人、不发 TCP。")
    else:
        cam_note = "无相机" if not args.with_cameras else "含相机"
        print(
            f"机器人 {args.robot_host}:{args.robot_port} "
            f"(timeout={args.robot_timeout_ms}ms, {cam_note})"
        )
        if args.reconnect:
            print(
                f"已开启自动重连（间隔 {args.reconnect_interval}s）；"
                "机械臂程序重启后本进程会等待并重连，无需重启 teleoperate。"
            )
            if not wait_for_robot(
                robot,
                args.robot_host,
                args.robot_port,
                args.reconnect_interval,
                first_attempt=True,
            ):
                teleop.disconnect()
                raise SystemExit("已取消（未连上机器人）。")
        else:
            print("正在连接机器人 ...")
            try:
                robot.connect()
            except ConnectionError as exc:
                teleop.disconnect()
                raise SystemExit(
                    f"机器人连接失败: {exc}\n提示: 确认机器人 TCP 已启动；"
                    "或去掉 --no-reconnect 以自动等待重连；仅测 VR 可加 --vr-only。"
                ) from exc
            print(f"已连接机器人，@ {args.fps}Hz 发送 SET_EE_DELTA actions。")

    print("XLeVR 遥操作运行中，Ctrl+C 停止。")
    print("右手 squeeze=粗调 | 前扳机=精调(无需 squeeze) | 摇杆 x=夹爪")
    print("左手摇杆：右=下一阶段，左=重录，上=机械臂复位，下=停止录制")
    print("未按 squeeze/扳机时机械臂不会动；下方每秒打印 enabled / squeeze / delta。")
    frame = 0
    robot_link_ok = robot is not None and robot.is_connected
    next_reconnect_at = 0.0
    status_keys = (
        "ee.enabled",
        "ee.delta_x",
        "ee.delta_y",
        "ee.delta_z",
        "ee.delta_rx",
        "ee.delta_ry",
        "ee.delta_rz",
        "gripper.pos",
    )

    def _print_action_status(prefix: str, action: dict) -> None:
        has_quat = raw_action.get("xlevr.orientation_quat") is not None
        has_pos = raw_action.get("xlevr.target_position") is not None
        vr_status = teleop.get_status()
        has_right = vr_status.get("has_right_goal")
        summary = ", ".join(f"{k}={action.get(k)}" for k in status_keys if k in action)
        buttons = raw_action.get("xlevr.buttons", {}) or {}
        summary = (
            f"squeeze={bool(buttons.get('squeeze', False))}, "
            f"trigger={raw_action.get('xlevr.trigger', 0.0)}, {summary}"
        )
        hint = ""
        if not has_right or not has_pos:
            hint = "  << 没有右手柄位姿。请在 VR 浏览器点 Enter VR，确认右手柄已配对；页面需强制刷新。"
        print(
            f"[{prefix} {frame}] right_goal={has_right} pos={has_pos} quat={has_quat} | {summary}{hint}",
            flush=True,
        )

    try:
        while True:
            start = time.perf_counter()

            if robot is not None:
                if robot_link_ok:
                    try:
                        obs = robot.get_observation()
                    except ROBOT_LINK_ERRORS as exc:
                        logging.warning("机器人连接断开（读状态）: %s", exc)
                        _safe_robot_disconnect(robot)
                        robot_link_ok = False
                        next_reconnect_at = 0.0
                        obs = fake_observation()
                        if not args.reconnect:
                            raise
                        print("机器人已断开，VR 仍运行；等待对端程序重启后自动重连 ...")
                else:
                    obs = fake_observation()
                    if args.reconnect and time.perf_counter() >= next_reconnect_at:
                        if try_robot_connect(robot, args.robot_host, args.robot_port):
                            robot_link_ok = True
                            print(f"机器人已重连 @ {args.robot_host}:{args.robot_port}")
                        else:
                            next_reconnect_at = time.perf_counter() + args.reconnect_interval
            else:
                obs = fake_observation()

            raw_action = teleop.get_action()
            ee_delta_action = teleop_action_processor((raw_action, obs))
            robot_action = robot_action_processor((ee_delta_action, obs))
            sent_action = None

            if robot is not None and robot_link_ok:
                try:
                    sent_action = robot.send_action(robot_action)
                except ROBOT_LINK_ERRORS as exc:
                    logging.warning("机器人连接断开（发动作）: %s", exc)
                    _safe_robot_disconnect(robot)
                    robot_link_ok = False
                    next_reconnect_at = 0.0
                    if not args.reconnect:
                        raise
                    print("机器人已断开，VR 仍运行；等待对端程序重启后自动重连 ...")
                if frame % args.print_every == 0:
                    _print_action_status("tcp", robot_action)
            elif robot is None and frame % args.print_every == 0:
                _print_action_status("vr-only", robot_action)

            if diagnostics_writer is not None:
                diagnostics = get_xlevr_diagnostics(teleop_action_processor)
                if diagnostics is not None:
                    diagnostics.update(
                        {
                            "recording_phase": "vr_only" if robot is None else "teleoperate",
                            "control_frame_index": frame,
                            "computed_robot_action": robot_action,
                            "sent_action": sent_action,
                        }
                    )
                    diagnostics_writer.write(diagnostics)

            frame += 1
            dt = time.perf_counter() - start
            precise_sleep(1 / args.fps - dt)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if diagnostics_writer is not None:
            diagnostics_writer.close()
        teleop.disconnect()
        if robot is not None and robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
