#!/usr/bin/env python
"""Phase 0: π0.5 推理服务在环开环验证。

目的:验证 π0.5 websocket 推理服务 + A10 真机闭环能以目标频率稳定运行,
为后续残差 RL 平台打基础。本脚本不含任何 RL,只是 π0.5 → A10 开环执行。

链路:
  A10Follower.get_observation() (关节 + 相机 RGB HWC)
    → 构建 π0.5 obs (state (1,7), images/right (3,224,224), images/top, prompt)
    → WebsocketClientPolicy.infer() → actions [T, 7] (绝对关节)
    → ChunkBuffer 逐步释放 a_vla[t]
    → A10Follower.send_action({joint_N.pos, gripper.pos})  [SET_JOINTS]

用法:
  # 先在 GPU 机器上起 π0.5 服务 (openpi 仓库):
  #   cd ~/Allen/openpi
  #   uv run scripts/serve_policy.py policy:checkpoint \
  #     --policy.config=pi05_a10_finetune \
  #     --policy.dir=checkpoints/pi05_a10_finetune/Reach_5_9_1/130000
  #
  # 再在机器人端跑本脚本 (lerobot conda env):
  #   python examples/dev_hil/phase0_pi05_openloop.py \
  #     --policy-host 127.0.0.1 --policy-port 8000 \
  #     --robot-host 192.168.1.12 --robot-port 8080 \
  #     --prompt "Reach the yellow lemon" --steps 50 --hz 10
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# 把 openpi-client 加入 sys.path (本仓库不直接依赖 openpi)
_OPENPI_CLIENT = "/home/allen/Allen/openpi/packages/openpi-client/src"
if _OPENPI_CLIENT not in sys.path:
    sys.path.insert(0, _OPENPI_CLIENT)

from openpi_client import websocket_client_policy  # noqa: E402

import cv2  # noqa: E402

from lerobot.robots.a10_follower.config_a10_follower import A10FollowerConfig  # noqa: E402
from lerobot.robots.a10_follower.a10_follower import A10Follower  # noqa: E402

logger = logging.getLogger(__name__)


class ChunkBuffer:
    """逐步释放 π0.5 action chunk;chunk 耗尽时重新推理。

    等价于 openpi_client.ActionChunkBroker,但内联实现以避免引入 ``tree`` 依赖。
    """

    def __init__(self, client: websocket_client_policy.WebsocketClientPolicy,
                 action_horizon: int) -> None:
        self._client = client
        self._horizon = action_horizon
        self._chunk: np.ndarray | None = None
        self._step = 0

    def reset(self) -> None:
        self._chunk = None
        self._step = 0

    def infer(self, obs: dict) -> dict:
        """返回当前步的 action dict (含 'actions' (7,));必要时先重新推理。"""
        if self._chunk is None:
            result = self._client.infer(obs)
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None, :]
            elif actions.ndim == 3 and actions.shape[0] == 1:
                actions = actions[0]
            self._chunk = np.ascontiguousarray(actions, dtype=np.float32)
            self._step = 0
        action = self._chunk[self._step]
        self._step += 1
        if self._step >= self._horizon or self._step >= len(self._chunk):
            self._chunk = None
        return {"actions": action}


def _resize_with_pad(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """与 openpi bridge 一致的等比缩放 + 居中 pad。"""
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _to_policy_chw(frame_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """RGB HWC → CHW uint8 (与 openpi USBCamera.preprocess_to_policy_chw 对齐)。"""
    frame = np.asarray(frame_rgb)
    if frame.ndim != 3 or frame.shape[-1] < 3 or frame.size == 0:
        frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        max_v = float(np.max(frame)) if frame.size > 0 else 0.0
        if max_v <= 1.01:
            frame = (np.clip(frame, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            frame = np.clip(frame, 0.0, 255.0).round().astype(np.uint8)
    else:
        frame = frame.astype(np.uint8, copy=False)
    frame = _resize_with_pad(frame, size, size)
    chw = np.transpose(frame, (2, 0, 1))
    return np.ascontiguousarray(chw, dtype=np.uint8)


def _build_observation(robot: A10Follower, prompt: str) -> dict:
    """从 A10Follower 观测构建 π0.5 期望的 obs dict。"""
    obs = robot.get_observation()
    # 关节状态:7D (joint_1..joint_6, gripper)
    state = np.array(
        [obs[f"{n}.pos"] for n in robot.joint_names], dtype=np.float32
    ).reshape(1, 7)

    # 相机:lerobot OpenCVCamera 输出 RGB HWC;π0.5 要 CHW 224²
    cam_keys = list(robot.cameras.keys())
    if len(cam_keys) == 0:
        right_chw = np.zeros((3, 224, 224), dtype=np.uint8)
        top_chw = right_chw
    else:
        right_key = cam_keys[0]
        right_rgb = obs[right_key]
        right_chw = _to_policy_chw(right_rgb, 224)
        if len(cam_keys) >= 2:
            top_chw = _to_policy_chw(obs[cam_keys[1]], 224)
        else:
            # 单相机:top 复用 right (与 openpi bridge 行为一致)
            top_chw = right_chw

    return {
        "observation/state": state,
        "observation/images/right": right_chw,
        "observation/images/top": top_chw,
        "prompt": prompt,
    }


def _action_to_robot_dict(action_7d: np.ndarray, joint_names: list[str]) -> dict:
    """π0.5 输出 7D [joint_1..joint_6, gripper_abs] → A10Follower joint 模式 dict。"""
    a = np.asarray(action_7d, dtype=np.float32).reshape(-1)
    out = {}
    for i, name in enumerate(joint_names):
        if i < len(a):
            out[f"{name}.pos"] = float(a[i])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0: π0.5 + A10 开环验证")
    parser.add_argument("--policy-host", type=str, default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--robot-host", type=str, default="192.168.1.12")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--prompt", type=str, default="Reach the yellow lemon")
    parser.add_argument("--steps", type=int, default=50, help="执行步数 (每步 1/hz 秒)")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--action-horizon", type=int, default=10, help="π0.5 chunk 长度")
    parser.add_argument("--camera-index", type=str, default=None,
                        help="覆盖默认相机索引 (如 0/2);不传用 A10FollowerConfig 默认")
    parser.add_argument("--dry-run", action="store_true",
                        help="不连机器人,只验证推理服务 + obs 构建")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- 机器人 ---
    cam = None
    if args.camera_index is not None:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.cameras.configs import Cv2Rotation
        cam = {"right_wrist_0_rgb": OpenCVCameraConfig(
            index_or_path=int(args.camera_index), fps=30, width=640, height=480,
            rotation=Cv2Rotation.ROTATE_180,
        )}
    robot = A10Follower(A10FollowerConfig(
        host=args.robot_host, port=args.robot_port,
        n_joints=7, use_ee_delta=False,  # 关节模式
        cameras=cam if cam is not None else None,
    ))

    # --- π0.5 客户端 + chunk broker ---
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.policy_host, port=args.policy_port,
    )
    broker = ChunkBuffer(client, args.action_horizon)

    if not args.dry_run:
        robot.connect()
        logger.info("A10 已连接,关节=%s", robot.joint_names)

    dt = 1.0 / args.hz
    infer_count = 0
    step_count = 0
    over_budget = 0  # 超时次数

    try:
        while step_count < args.steps:
            t0 = time.perf_counter()

            obs = _build_observation(robot, args.prompt)
            t_infer = time.perf_counter()
            result = broker.infer(obs)
            infer_ms = (time.perf_counter() - t_infer) * 1e3
            infer_count += 1

            action = result["actions"]  # (7,) 当前步
            action_dict = _action_to_robot_dict(action, robot.joint_names)

            if not args.dry_run:
                robot.send_action(action_dict)

            step_count += 1
            elapsed = time.perf_counter() - t0
            remain = dt - elapsed
            if remain < 0:
                over_budget += 1

            if step_count % 10 == 0 or step_count <= 3:
                logger.info(
                    "step=%d infer_ms=%.1f total_ms=%.1f action=%s",
                    step_count, infer_ms, elapsed * 1e3,
                    np.array2string(action, precision=4, suppress_small=True),
                )

            if remain > 0:
                time.sleep(remain)

    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C,停止。")
    finally:
        if not args.dry_run and robot.is_connected:
            robot.disconnect()
        client.close()

    logger.info(
        "完成: steps=%d infer_calls=%d 超时次数=%d (%.1f%%)",
        step_count, infer_count, over_budget,
        100.0 * over_budget / max(1, step_count),
    )
    if over_budget / max(1, step_count) > 0.1:
        logger.warning("超时比例 >10%%,π0.5 推理跟不上 %gHz,需降频或换更小策略。", args.hz)
    else:
        logger.info("验收通过:π0.5 + A10 开环在 %gHz 稳定。", args.hz)


if __name__ == "__main__":
    main()
