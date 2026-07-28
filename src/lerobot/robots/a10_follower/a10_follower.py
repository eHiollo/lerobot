
import logging
import time
from typing import Any
import numpy as np
from functools import cached_property
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.cameras.utils import make_cameras_from_configs

from lerobot.robots import Robot
from .config_a10_follower import A10FollowerConfig
from .a10_client import A10TCPClient

logger = logging.getLogger(__name__)

class A10Follower(Robot):
    config_class = A10FollowerConfig
    name = "a10_follower"

    def __init__(self, config: A10FollowerConfig):
        super().__init__(config)
        self.config = config
        
        if self.config.n_joints == 7:
             self.joint_names = [
                "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"
            ]
        elif self.config.n_joints == 6:
             self.joint_names = [
                "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
            ]
        else:
             self.joint_names = [f"joint_{i}" for i in range(self.config.n_joints)]

        self.client = A10TCPClient(
            host=config.host,
            port=config.port,
            joint_names=self.joint_names,
            timeout_ms=config.timeout_ms
        )
        
        # Initialize cameras from config (standard LeRobot way)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            name: (cam.height, cam.width, 3) for name, cam in self.cameras.items()
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        feats = {**self._motors_ft, **self._cameras_ft}
        if self.config.use_ee_target:
            feats.update(
                {
                    "ee.x": float,
                    "ee.y": float,
                    "ee.z": float,
                    "ee.rx": float,
                    "ee.ry": float,
                    "ee.rz": float,
                }
            )
        return feats

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.config.use_ee_target:
            return {
                "ee.enabled": bool,
                "ee.target_x": float,
                "ee.target_y": float,
                "ee.target_z": float,
                "ee.target_rx": float,
                "ee.target_ry": float,
                "ee.target_rz": float,
                "gripper.pos": float,
            }
        if self.config.use_ee_delta:
            return {
                "ee.enabled": bool,
                "ee.delta_x": float,
                "ee.delta_y": float,
                "ee.delta_z": float,
                "ee.delta_rx": float,
                "ee.delta_ry": float,
                "ee.delta_rz": float,
                "gripper.pos": float,
            }
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.client.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            logger.info(f"{self} already connected.")
            return
        
        self.client.connect()
        for cam in self.cameras.values():
            cam.connect()
        # logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        self.client.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

    # ---------- 标定 / 配置（先空着） ----------

    @property
    def is_calibrated(self) -> bool:
        return True

    def setup_motors(self) -> None:
        pass
    
    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ---------- 观测 / 动作 ----------

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        start = time.perf_counter()
        state = self.client.get_observation()  # {"q": ...}

        q = state["q"]
        obs_dict = {}
        for i, name in enumerate(self.joint_names):
            if i < len(q):
                obs_dict[f"{name}.pos"] = float(q[i])

        # target 模式下额外取末端位姿，供 VR 端"原点增量"处理器抓取 robot_origin。
        if self.config.use_ee_target:
            ee_state = self.client.get_ee_state()
            ee = ee_state.get("ee")
            if ee is not None:
                obs_dict["ee.x"] = float(ee[0])
                obs_dict["ee.y"] = float(ee[1])
                obs_dict["ee.z"] = float(ee[2])
                obs_dict["ee.rx"] = float(ee[3])
                obs_dict["ee.ry"] = float(ee[4])
                obs_dict["ee.rz"] = float(ee[5])

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # 2. Get Images from Local Cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read(timeout_ms=1000)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.use_ee_target and "ee.target_x" in action:
            enabled = bool(action.get("ee.enabled", False))
            if enabled:
                arm = [
                    float(action.get("ee.target_x", 0.0)),
                    float(action.get("ee.target_y", 0.0)),
                    float(action.get("ee.target_z", 0.0)),
                    float(action.get("ee.target_rx", 0.0)),
                    float(action.get("ee.target_ry", 0.0)),
                    float(action.get("ee.target_rz", 0.0)),
                ]
            else:
                # 未按下 squeeze：保持上一次目标(由处理器填入 ee.target_* = last target)
                arm = [
                    float(action.get("ee.target_x", 0.0)),
                    float(action.get("ee.target_y", 0.0)),
                    float(action.get("ee.target_z", 0.0)),
                    float(action.get("ee.target_rx", 0.0)),
                    float(action.get("ee.target_ry", 0.0)),
                    float(action.get("ee.target_rz", 0.0)),
                ]

            actions = arm + [float(action.get("gripper.pos", 0.0))]
            self.client.send_ee_target(actions)
            return action

        if self.config.use_ee_delta and "ee.delta_x" in action:
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

            actions = arm + [float(action.get("gripper.pos", 0.0))]
            self.client.send_ee_delta(actions)
            return action

        q_target = []
        for name in self.joint_names:
            key = f"{name}.pos"
            if key in action:
                q_target.append(action[key])
            else:
                q_target.append(0.0)

        q_target_arr = np.array(q_target, dtype=np.float32)
        self.client.send_action(q_target_arr)

        return action
