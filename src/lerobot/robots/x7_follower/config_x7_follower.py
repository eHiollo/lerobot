from dataclasses import dataclass, field
from typing import Dict

from lerobot.robots.robot import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

@RobotConfig.register_subclass("x7_follower")
@dataclass
class X7FollowerConfig(RobotConfig):
    host: str = "127.0.0.1"
    port: int = 8000
    n_joints: int = 8
    
    # 默认相机配置，可以被命令行覆盖
    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "base_0_rgb": OpenCVCameraConfig(index_or_path=4, fps=30, width=640, height=480),
            # "left_wrist_0_rgb": OpenCVCameraConfig(index_or_path=1, fps=30, width=640, height=480),
        }
    )

    # 允许的最大相对移动量（安全限制）
    max_relative_target: float | None = None

