import logging
import time
from typing import Any
import numpy as np
from functools import cached_property
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.cameras.utils import make_cameras_from_configs

from lerobot.robots import Robot
from .config_x7_follower import X7FollowerConfig
from .x7_client import X7TCPClient

logger = logging.getLogger(__name__)

class X7Follower(Robot):
    config_class = X7FollowerConfig
    name = "x7_follower"

    def __init__(self, config: X7FollowerConfig):
        super().__init__(config)
        self.config = config
        
        # 8 DOF: 7 joints + 1 gripper
        if self.config.n_joints == 8:
             self.joint_names = [
                "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"
            ]
        else:
             self.joint_names = [f"joint_{i}" for i in range(self.config.n_joints)]

        self.client = X7TCPClient(
            host=config.host,
            port=config.port,
            joint_names=self.joint_names
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
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.client.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
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

    # ---------- 标定 / 配置 ----------

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
        
        # 1. Get Motor State
        start = time.perf_counter()
        state = self.client.get_state()  # {"q": ...}
        
        q = state["q"]
        obs_dict = {}
        for i, name in enumerate(self.joint_names):
            if i < len(q):
                obs_dict[f"{name}.pos"] = float(q[i])

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # 2. Get Images from Local Cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
            
        # Extract joint positions in order
        q_target = []
        for name in self.joint_names:
            key = f"{name}.pos"
            if key in action:
                q_target.append(action[key])
            else:
                # 如果动作中缺少某个关节，默认保持 0 或者上一次的位置？
                # 这里简单给 0，或者应该从 self.client._last_q 获取
                # 为了安全，最好是保持当前位置，但这里简化处理
                q_target.append(0.0) 
        
        q_target_arr = np.array(q_target, dtype=np.float32)
        self.client.send_action(q_target_arr)
        
        return action
