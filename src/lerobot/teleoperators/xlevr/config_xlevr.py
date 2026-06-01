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

    # VR position/orientation delta scaling applied before sending to robot
    pos_scale: float = 1.0
    angle_scale: float = 1.0
    vr_to_robot_scale: float = 1.0

    # Control loop rate — should match robot TCP / lerobot-record fps (default 15Hz)
    control_fps: int = 15

    # Ignore tiny deltas below these thresholds (reduce noise at control rate)
    pos_deadzone_m: float = 0.0005
    angle_deadzone_deg: float = 0.05
    # Per-frame delta caps; None = no limit (full VR follow). Set e.g. 0.05 / 10.0 to tame jitter.
    max_delta_pos_m: float | None = None
    max_delta_angle_deg: float | None = None

    # Hold squeeze (side grip) to enable arm motion
    require_squeeze_to_move: bool = True

    # Gripper: pass through right thumbstick x directly (typically [-1, 1])
    gripper_thumbstick_axis: str = "x"
    gripper_thumbstick_deadzone: float = 0.05

    # VR frame: +X right, +Y up, +Z backward
    # Robot frame: +X forward, +Y left, +Z up
    #   forward = -VR_Z, left = -VR_X, up = +VR_Y
    axis_remap: tuple[tuple[float, float, float], ...] = field(
        default_factory=lambda: (
            (0.0, 0.0, -1.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    # VR wrist: roll~Z, pitch~X, yaw~Y  ->  robot roll~X, pitch~Y, yaw~Z
    delta_roll_sign: float = -1.0
    delta_pitch_sign: float = 1.0
    delta_yaw_sign: float = -1.0

    # Use left-controller thumbstick for dataset/recording events
    enable_left_events: bool = True
    thumbstick_event_threshold: float = 0.7
