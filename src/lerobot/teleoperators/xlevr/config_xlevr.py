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

    # A1.1 input-guard mode. Set to "legacy_frame_delta" only to recover the
    # pre-A1 input behavior for comparison and rollback.
    position_control_mode: str = "safe_frame_delta"
    stale_timeout_s: float = 0.25
    max_vr_speed_m_s: float = 2.0
    spike_recovery_frames: int = 2

    # A2.1 computes an anchored-pose command for diagnostics only. It does not
    # change the current RobotAction or send the future SET_EE_ANCHOR protocol.
    compute_anchor_shadow: bool = True

    # Motion shaping belongs to the validated A10 ``vr_vel`` 500 Hz controller.
    # ``lerobot_a1`` keeps the former A1.2 implementation as an explicit
    # rollback/experiment mode; do not combine it with normal vr_vel shaping
    # unless double filtering is intentionally being tested.
    motion_shaping_mode: str = "robot_controller"
    position_cutoff_hz: float = 5.0
    max_linear_speed_m_s: float = 0.25
    max_linear_accel_m_s2: float = 1.0
    engage_ramp_s: float = 0.35

    # A1.3 diagnostics are written as a JSONL sidecar by lerobot_record. They
    # never become training action features and never enter the A10 wire payload.
    record_vr_diagnostics: bool = True
    diagnostics_queue_size: int = 2048
    diagnostics_flush_every: int = 30

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
