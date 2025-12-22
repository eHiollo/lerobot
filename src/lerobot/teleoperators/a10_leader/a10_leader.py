#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots.a10_follower.a10_client import A10TCPClient

from ..teleoperator import Teleoperator
from .config_a10_leader import A10LeaderConfig

logger = logging.getLogger(__name__)


class A10Leader(Teleoperator):
    """
    A10 Leader Arm connected via TCP.
    """

    config_class = A10LeaderConfig
    name = "a10_leader"

    def __init__(self, config: A10LeaderConfig):
        super().__init__(config)
        self.config = config
        
        self.motor_names = [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6","gripper"
        ]
        
        self.client = A10TCPClient(
            host=config.host, 
            port=config.port,
            joint_names=self.motor_names
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.client.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            logger.info(f"{self} already connected.")
            return

        self.client.connect()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.client.disconnect()
        logger.info(f"{self} disconnected.")



    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        state_data = self.client.get_state()   # {"q": np.ndarray([...])}
        q = state_data["q"]

        # 安全检查：确保配置和控制器返回的维度一致
        if len(q) != len(self.motor_names):
            raise ValueError(
                f"Expected {len(self.motor_names)} joints, but got {len(q)} from controller"
            )

        # motor_names: ["joint_1", "joint_2", ...]
        action = {f"{name}.pos": float(val) for name, val in zip(self.motor_names, q)}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action


    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Implement force feedback if supported by A10 leader
        pass
