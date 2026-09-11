#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.xlevr.quaternion_utils import AXIS_REMAP_VR_TO_ROBOT


@TeleoperatorConfig.register_subclass("xlevr")
@dataclass
class XLeVRTeleopConfig(TeleoperatorConfig):
    """Configuration for XLeVR WebXR teleoperation."""

    xlevr_path: str = "/home/robot/VLA/XLeRobot/XLeVR"
    arm: str = "right"
    connection_timeout: float = 10.0

    pos_scale: float = 0.9
    angle_scale: float = 1.3
    vr_to_robot_scale: float = 1.0

    fine_trigger_threshold: float = 0.5
    fine_scale_factor: float = 0.5

    control_fps: int = 30

    pos_deadzone_m: float = 0.0005
    angle_deadzone_deg: float = 0.05
    max_delta_pos_m: float | None = None
    max_delta_angle_deg: float | None = None

    require_squeeze_to_move: bool = True

    gripper_thumbstick_axis: str = "x"
    gripper_thumbstick_deadzone: float = 0.05

    axis_remap: tuple[tuple[float, float, float], ...] = field(
        default_factory=lambda: AXIS_REMAP_VR_TO_ROBOT
    )

    enable_left_events: bool = True
    thumbstick_event_threshold: float = 0.7
