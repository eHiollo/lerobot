#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("xlevr")
@dataclass
class XLeVRTeleopConfig(TeleoperatorConfig):
    """Configuration for XLeVR WebXR teleoperation."""

    xlevr_path: str = "/home/allen/Allen/XLeRobot/XLeVR"
    arm: str = "right"
    connection_timeout: float = 10.0

    pos_scale: float = 0.9
    angle_scale: float = 1.3
    vr_to_robot_scale: float = 1.0

    fine_trigger_threshold: float = 0.5
    fine_scale_factor: float = 0.5

    control_fps: int = 30

    # True = 闭环"原点增量"绝对目标(SET_EE_TARGET, 零漂移, 需机器人 use_ee_target)；
    # False = 帧间增量(SET_EE_DELTA, 累加语义)。
    use_ee_target_mode: bool = False

    # One Euro 滤波参数(仅 target 模式生效)。
    enable_filter: bool = True
    filter_min_cutoff: float = 1.0
    filter_beta: float = 0.007

    pos_deadzone_m: float = 0.0005
    angle_deadzone_deg: float = 0.05
    max_delta_pos_m: float | None = None
    max_delta_angle_deg: float | None = None

    require_squeeze_to_move: bool = True

    gripper_thumbstick_axis: str = "x"
    gripper_thumbstick_deadzone: float = 0.05

    # VR body: +X right, +Y up, +Z back. Robot: +X up, +Y right, +Z forward.
    axis_remap: tuple[tuple[float, float, float], ...] = field(
        default_factory=lambda: (
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
        )
    )

    enable_left_events: bool = True
    thumbstick_event_threshold: float = 0.7
