import logging
import time
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots.x7_follower.x7_client import X7TCPClient

from ..teleoperator import Teleoperator
from .config_x7_leader import X7LeaderConfig

logger = logging.getLogger(__name__)


class X7Leader(Teleoperator):
    """
    X7 Leader Arm connected via TCP.
    """

    config_class = X7LeaderConfig
    name = "x7_leader"

    def __init__(self, config: X7LeaderConfig):
        super().__init__(config)
        self.config = config
        
        if self.config.n_joints == 8:
             self.motor_names = [
                "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"
            ]
        else:
             self.motor_names = [f"joint_{i}" for i in range(self.config.n_joints)]
        
        self.client = X7TCPClient(
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
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.client.connect()
        # logger.info(f"{self} connected.")

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
        
        state_data = self.client.get_state()
        q = state_data["q"]
        
        action = {}
        for i, name in enumerate(self.motor_names):
            if i < len(q):
                action[f"{name}.pos"] = float(q[i])

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass
