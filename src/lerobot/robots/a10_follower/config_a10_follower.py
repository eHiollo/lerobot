# config_a10_robot.py
from dataclasses import dataclass, field
from typing import Dict
from lerobot.cameras.configs import Cv2Rotation
from lerobot.robots import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

@RobotConfig.register_subclass("a10_follower")
@dataclass
class A10FollowerConfig(RobotConfig):
    host: str = "192.168.1.105"
    port: int = 8080
    n_joints: int = 7
    timeout_ms: int = 5000  

    # 默认相机配置，可以被命令行覆盖
    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            #"base_0_rgb": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480),
            "right_wrist_0_rgb": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480,rotation=Cv2Rotation.ROTATE_180),
        }
    )

