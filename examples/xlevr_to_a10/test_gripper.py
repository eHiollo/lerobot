#!/usr/bin/env python3
"""
A10 夹爪直连测试：绕过 VR 和 lerobot_record，直接往 A10 发夹爪命令。

用途：把「夹爪不动」的故障定位到 A10 侧还是 VR/LeRobot 侧。
本脚本只发前 6 维为 0 的 EE 增量（机械臂不动）+ 第 7 维夹爪指令，
因此机械臂不会动，只考验夹爪这一条链路。

读回原理（因此本脚本能自己得出结论，不需要人盯着看）：
  A10 main.cpp 的 state_update_thread 把 13 维 q 写入 robot_q_，其中
  [12] = a10_tcp::g_vr_grip_actual_mm（夹爪开度 mm）。
  TCP 的 GET_FOLLOWER_STATE 实际返回 7 维：[j0..j5, 夹爪mm]（见 send_follower_state），
  即夹爪在下标 6。服务端只在收到查询时响应、不主动推流，所以读回干净可靠。

判据：
  * 读回值随命令变化（张开升高 / 闭合降低）-> A10 夹爪链路正常，
    问题在 VR/LeRobot 侧，或只是“推得不够久”。
  * 读回值恒为 0 且完全不变 -> A10 侧未驱动夹爪。最可能是启动时
    /dev/ttyUSB0 未连上进入 arm-only 模式，夹爪命令被全部忽略。

用法：
    python examples/xlevr_to_a10/test_gripper.py
    python examples/xlevr_to_a10/test_gripper.py --robot-host 192.168.110.124 --robot-port 8080
    python examples/xlevr_to_a10/test_gripper.py --hold-s 4
    python examples/xlevr_to_a10/test_gripper.py --dry-run   # 只打印不发

夹爪语义（A10 侧 a10_gripper_bridge.hpp）：
    +1 张开，-1 闭合，死区 0.05，行程 0..100 mm，默认速度 grip_vel=50 mm/s
    注意这是「速度」指令（会被积分），所以必须持续推住才会走完行程。
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from lerobot.robots.a10_follower.a10_client import A10TCPClient
from lerobot.utils.robot_utils import precise_sleep

# A10 侧 k_gripper_mm_min=0(闭) / k_gripper_mm_max=100(开)，速度默认 50 mm/s。
GRIP_CLOSE = -1.0
GRIP_OPEN = 1.0
GRIP_HOLD = 0.0
GRIPPER_IDX = 6  # GET_FOLLOWER_STATE 返回 [j0..j5, 夹爪mm]，夹爪在下标 6


def make_actions(grip: float) -> list[float]:
    """6 维 EE 增量置零（机械臂不动）+ 夹爪指令。"""
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(grip)]


def read_gripper_mm(client: A10TCPClient, label: str) -> float | None:
    """读回夹爪开度(mm)。失败返回 None。"""
    try:
        obs = client.get_observation()
    except Exception as exc:  # noqa: BLE001
        print(f"  [{label}] 读取失败: {exc}")
        return None
    q = np.asarray(obs["q"], dtype=float).ravel()
    if q.size <= GRIPPER_IDX:
        print(f"  [{label}] q 维度不足({q.size})，无法取 [12]；原始 q={q.tolist()}")
        return None
    joints = ", ".join(f"{v:+.3f}" for v in q[:6])
    print(f"  [{label}] 夹爪={q[GRIPPER_IDX]:7.2f} mm | 关节=[{joints}]")
    return float(q[GRIPPER_IDX])


def hold(client: A10TCPClient | None, grip: float, seconds: float, fps: int, label: str) -> None:
    n = max(1, int(round(seconds * fps)))
    actions = make_actions(grip)
    payload = json.dumps({"actions": actions}, separators=(",", ":"))
    print(f"  {label:12} grip={grip:+.2f}  x{n}帧 ({seconds:.1f}s)  -> SET_EE_DELTA {payload}", flush=True)
    dt = 1.0 / fps
    for _ in range(n):
        t0 = time.perf_counter()
        if client is not None:
            client.send_ee_delta(actions)
        precise_sleep(dt - (time.perf_counter() - t0))


def main() -> int:
    ap = argparse.ArgumentParser(description="A10 夹爪诊断（直驱 + 位置读回）")
    ap.add_argument("--robot-host", default="192.168.110.124")
    ap.add_argument("--robot-port", type=int, default=8080)
    ap.add_argument("--fps", type=int, default=30, help="发送频率，应与录制一致")
    ap.add_argument("--hold-s", type=float, default=4.0,
                    help="每个方向持续秒数。必须够长：grip_vel 默认 50mm/s，100mm 行程需约 2s")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不连接不发送")
    args = ap.parse_args()

    print("=" * 76)
    print("A10 夹爪诊断（绕过 VR，直驱 + 位置读回）")
    print(f"  目标: {args.robot_host}:{args.robot_port}  fps={args.fps}  每方向 {args.hold_s:.1f}s")
    print("  语义: -1=闭合(0mm)  +1=张开(100mm)  速度约 50mm/s（grip_vel 默认值）")
    print("  安全: 前 6 维 EE 增量恒为 0，机械臂不动")
    print("=" * 76)

    client: A10TCPClient | None = None
    if args.dry_run:
        print("[dry-run] 不连接、不发送\n")
    else:
        client = A10TCPClient(args.robot_host, args.robot_port)
        try:
            client.connect()
        except Exception as exc:  # noqa: BLE001
            print(f"\n[FAIL] 连接 {args.robot_host}:{args.robot_port} 失败: {exc}")
            print("       请确认 A10 控制器已启动、网络可达、端口未被占用。")
            return 2
        print(f"[OK] 已连接 {args.robot_host}:{args.robot_port}\n")

    if client is None:
        hold(None, GRIP_OPEN, args.hold_s, args.fps, "张开 OPEN")
        hold(None, GRIP_CLOSE, args.hold_s, args.fps, "闭合 CLOSE")
        print("\n[dry-run 结束]")
        return 0

    base = after_open = after_close = None
    try:
        print("[基线]")
        base = read_gripper_mm(client, "base")

        print("\n[第 1 步] 张开")
        hold(client, GRIP_OPEN, args.hold_s, args.fps, "张开 OPEN")
        hold(client, GRIP_HOLD, 0.4, args.fps, "停 INPUT0")
        after_open = read_gripper_mm(client, "after_open")

        print("\n[第 2 步] 闭合")
        hold(client, GRIP_CLOSE, args.hold_s, args.fps, "闭合 CLOSE")
        hold(client, GRIP_HOLD, 0.4, args.fps, "停 INPUT0")
        after_close = read_gripper_mm(client, "after_close")
    except KeyboardInterrupt:
        print("\n[中断] 用户中止")
        return 1
    finally:
        try:
            client.disconnect()
        except Exception:  # noqa: BLE001
            pass

    # ---------------- 判定 ----------------
    print()
    print("=" * 76)
    print("判定：")
    vals = [v for v in (base, after_open, after_close) if v is not None]
    if not vals:
        print("  [无法判定] 没读到夹爪位置。请确认 A10 在跑、端口正确。")
        return 3

    span = max(vals) - min(vals)
    if after_open is not None and after_close is not None and span >= 1.0:
        print(f"  [有响应] 开度变化 {span:.2f} mm"
              f" (基线={base} / 张开后={after_open} / 闭合后={after_close})")
        print("  => A10 侧夹爪链路正常。问题在 VR/LeRobot 侧，或只是“推摇杆时间太短”。")
        print("     注意 A10 夹爪是速度指令：需持续推住 1~3 秒，轻推一下只动约 1.6mm。")
        print("=" * 76)
        return 0

    print(f"  [无响应] 读数始终 ~{vals[0]:.2f} mm（变化仅 {span:.2f} mm）")
    print("  => A10 侧没有驱动夹爪。最可能原因：")
    print("     1) 启动时 /dev/ttyUSB0 没连上 -> arm-only 模式，夹爪命令被全部忽略。")
    print("        查 A10 启动日志：应为 'Gripper: connected (USB)'，")
    print("        若是 'Gripper: not detected, arm-only mode' 即为此问题。")
    print("     2) 舵机未供电 / 机械卡死 / 舵机 id 不是 10。")
    print("     3) 在 A10 机器上直接试 aris 命令：g_status / g_open / g_close。")
    print("=" * 76)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
