# !/usr/bin/env python
"""A10 关节空间 gym 环境,适配 HIL-SERL 残差 RL 平台。

与 ``gym_manipulator.RobotEnv`` (SO100, 依赖 ``robot.bus``) 并列,不侵入 SO100 路径。
动作空间 7D [-1,1] 归一化关节目标,env 内反归一化为绝对关节送 ``SET_JOINTS``,
与 π0.5 输出 (绝对关节 7D) 对齐,残差头加 Δa 后送本 env。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)

# A10 默认关节限位 (degree),gripper 为 mm (0-100)。真机标定后可覆盖。
DEFAULT_JOINT_LOWER = np.array([-170, -90, -90, -90, -90, -170, 0.0], dtype=np.float32)
DEFAULT_JOINT_UPPER = np.array([170, 90, 90, 90, 90, 170, 100.0], dtype=np.float32)


def _denormalize(action_norm: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """[-1,1] -> [lower, upper]。"""
    return (action_norm + 1.0) * 0.5 * (upper - lower) + lower


class A10RobotEnv(gym.Env):
    """A10 关节空间 gym 环境 (不依赖 ``robot.bus``)。"""

    def __init__(
        self,
        robot,
        display_cameras: bool = False,
        reset_pose: list[float] | None = None,
        reset_time_s: float = 5.0,
        joint_lower: list[float] | None = None,
        joint_upper: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.robot = robot
        self.display_cameras = display_cameras

        if not self.robot.is_connected:
            self.robot.connect()

        self._joint_names = list(self.robot.joint_names)
        self._image_keys = list(self.robot.cameras.keys())

        self.joint_lower = np.asarray(joint_lower if joint_lower is not None else DEFAULT_JOINT_LOWER, dtype=np.float32)
        self.joint_upper = np.asarray(joint_upper if joint_upper is not None else DEFAULT_JOINT_UPPER, dtype=np.float32)
        assert self.joint_lower.shape == (7,) and self.joint_upper.shape == (7,)

        self.reset_pose = np.asarray(reset_pose, dtype=np.float32) if reset_pose is not None else None
        self.reset_time_s = reset_time_s
        if self.reset_pose is not None:
            assert self.reset_pose.shape == (7,), f"reset_pose shape {self.reset_pose.shape} != (7,)"
            if np.any(self.reset_pose < self.joint_lower) or np.any(self.reset_pose > self.joint_upper):
                raise ValueError(
                    f"reset_pose {self.reset_pose.tolist()} 超出关节限位 "
                    f"[{self.joint_lower.tolist()}, {self.joint_upper.tolist()}]"
                )
        self.current_step = 0
        self._raw_joint_positions: dict[str, float] | None = None
        self._setup_spaces()

    def _setup_spaces(self) -> None:
        obs = self._get_observation()
        spaces: dict[str, gym.Space] = {}
        for key in self._image_keys:
            img = obs["pixels"][key]
            spaces[f"{OBS_IMAGES}.{key}"] = gym.spaces.Box(0, 255, img.shape, np.uint8)
        spaces[OBS_STATE] = gym.spaces.Box(-10.0, 10.0, obs["agent_pos"].shape, np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        # 动作:7D [-1,1] 归一化关节目标
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)

    def _get_observation(self) -> dict[str, Any]:
        raw = self.robot.get_observation()
        joints = np.array([raw[f"{n}.pos"] for n in self._joint_names], dtype=np.float32)
        pixels = {k: raw[k] for k in self._image_keys}
        self._raw_joint_positions = {f"{n}.pos": float(raw[f"{n}.pos"]) for n in self._joint_names}
        return {"agent_pos": joints, "pixels": pixels, **self._raw_joint_positions}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        start = time.perf_counter()
        if self.reset_pose is not None:
            self._smooth_reset(self.reset_pose)
        precise_sleep(max(0.0, self.reset_time_s - (time.perf_counter() - start)))
        super().reset(seed=seed)
        self.current_step = 0
        obs = self._get_observation()
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def _smooth_reset(self, target: np.ndarray) -> None:
        """用 SET_JOINTS 平滑插值回 reset_pose (关节空间,绕过 EE delta 累积漂移)。"""
        cur = np.array(
            [self.robot.get_observation()[f"{n}.pos"] for n in self._joint_names],
            dtype=np.float32,
        )
        for pose in np.linspace(cur, target, 50):
            self.robot.send_action({f"{n}.pos": float(p) for n, p in zip(self._joint_names, pose)})
            precise_sleep(0.015)

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape == (7,), f"action shape {action.shape} != (7,)"
        # 防御性 clip:确保归一化动作在 [-1,1],反归一化后才不会超出关节限位
        action = np.clip(action, -1.0, 1.0)
        abs_joints = _denormalize(action, self.joint_lower, self.joint_upper)
        self.robot.send_action({f"{n}.pos": float(v) for n, v in zip(self._joint_names, abs_joints)})
        obs = self._get_observation()
        if self.display_cameras:
            self.render()
        self.current_step += 1
        return obs, 0.0, False, False, {TeleopEvents.IS_INTERVENTION: False}

    def render(self) -> None:
        import cv2
        obs = self._get_observation()
        for key in self._image_keys:
            img = obs["pixels"][key]
            if hasattr(img, "numpy"):
                img = img.numpy()
            cv2.imshow(key, cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def close(self) -> None:
        if self.robot.is_connected:
            self.robot.disconnect()

    def get_raw_joint_positions(self) -> dict[str, float]:
        return self._raw_joint_positions or {}


def make_a10_robot_env(cfg) -> tuple[gym.Env, Teleoperator | None]:
    """从 HILSerlRobotEnvConfig 构建 A10 真机环境 + teleop 设备。

    Returns:
        (env, teleop_device);teleop 为 None 表示无干预设备 (Phase 0/1 验证用)。
    """
    from lerobot.robots import make_robot_from_config
    from lerobot.teleoperators import make_teleoperator_from_config

    robot = make_robot_from_config(cfg.robot)
    teleop_device = None
    if cfg.teleop is not None:
        teleop_device = make_teleoperator_from_config(cfg.teleop)
        teleop_device.connect()

    reset_pose = None
    reset_time_s = 5.0
    if cfg.processor.reset is not None:
        reset_pose = cfg.processor.reset.fixed_reset_joint_positions
        reset_time_s = cfg.processor.reset.reset_time_s

    display_cameras = (
        cfg.processor.observation.display_cameras if cfg.processor.observation is not None else False
    )

    env = A10RobotEnv(
        robot=robot,
        display_cameras=display_cameras,
        reset_pose=reset_pose,
        reset_time_s=reset_time_s,
    )
    return env, teleop_device


def make_a10_processors(env, teleop_device, cfg, device: str = "cpu"):
    """A10 关节空间处理器管线 (最小版,无 IK 链)。

    Phase 1:仅观测处理 + time limit + (可选) reward classifier。
    Phase 2 将补干预处理 (JointInterventionProcessorStep)。
    """
    from lerobot.processor import (
        AddBatchDimensionProcessorStep,
        DataProcessorPipeline,
        DeviceProcessorStep,
        ImageCropResizeProcessorStep,
        RewardClassifierProcessorStep,
        TimeLimitProcessorStep,
        VanillaObservationProcessorStep,
    )
    from lerobot.processor.converters import identity_transition

    terminate_on_success = (
        cfg.processor.reset.terminate_on_success if cfg.processor.reset is not None else True
    )

    env_steps = [VanillaObservationProcessorStep()]
    if cfg.processor.image_preprocessing is not None:
        env_steps.append(ImageCropResizeProcessorStep(
            crop_params_dict=cfg.processor.image_preprocessing.crop_params_dict,
            resize_size=cfg.processor.image_preprocessing.resize_size,
        ))
    if cfg.processor.reset is not None:
        env_steps.append(TimeLimitProcessorStep(
            max_episode_steps=int(cfg.processor.reset.control_time_s * cfg.fps)
        ))
    if (cfg.processor.reward_classifier is not None
            and cfg.processor.reward_classifier.pretrained_path is not None):
        env_steps.append(RewardClassifierProcessorStep(
            pretrained_path=cfg.processor.reward_classifier.pretrained_path,
            device=device,
            success_threshold=cfg.processor.reward_classifier.success_threshold,
            success_reward=cfg.processor.reward_classifier.success_reward,
            terminate_on_success=terminate_on_success,
        ))
    env_steps.append(AddBatchDimensionProcessorStep())
    env_steps.append(DeviceProcessorStep(device=device))

    # Phase 2 TODO: 补 JointInterventionProcessorStep (teleop_action 7D 关节覆盖)
    action_steps: list = []

    return (
        DataProcessorPipeline(steps=env_steps, to_transition=identity_transition, to_output=identity_transition),
        DataProcessorPipeline(steps=action_steps, to_transition=identity_transition, to_output=identity_transition),
    )
