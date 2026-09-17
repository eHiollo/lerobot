#!/usr/bin/env python

import copy
import math
import uuid
from dataclasses import dataclass, field

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotAction, RobotActionProcessorStep
from lerobot.teleoperators.xlevr.quaternion_utils import (
    AXIS_REMAP_VR_TO_ROBOT,
    delta_position_body,
    parse_quat_xyzw,
    quat_delta_rotvec_body_rad,
    remap_position,
    remap_rotvec,
)


class _SecondOrderLowPass:
    """Dependency-free two-pole low-pass for a 3D velocity signal."""

    def __init__(self, cutoff_hz: float):
        self.cutoff_hz = cutoff_hz
        self.reset()

    def reset(self) -> None:
        self._stage_1 = np.zeros(3, dtype=float)
        self._stage_2 = np.zeros(3, dtype=float)

    def update(self, value: np.ndarray, dt_s: float) -> np.ndarray:
        if self.cutoff_hz <= 0.0:
            return value.copy()
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * dt_s)
        self._stage_1 += alpha * (value - self._stage_1)
        self._stage_2 += alpha * (self._stage_1 - self._stage_2)
        return self._stage_2.copy()


@ProcessorStepRegistry.register("xlevr_to_ee_delta")
@dataclass
class XLeVRDeltaEEMapper(RobotActionProcessorStep):
    """
    Convert consecutive XLeVR poses into body-frame deltas, then remap to A10 EE-local axes.

    Orientation: quaternion frame delta (body) -> rotvec (rad) -> SET_EE_DELTA[3:6].
    """

    pos_scale: float = 0.9
    angle_scale: float = 1.3
    vr_to_robot_scale: float = 1.0
    fine_trigger_threshold: float = 0.5
    fine_scale_factor: float = 0.5
    pos_deadzone_m: float = 0.0005
    angle_deadzone_deg: float = 0.05
    max_delta_pos_m: float | None = None
    max_delta_angle_deg: float | None = None
    position_control_mode: str = "safe_frame_delta"
    stale_timeout_s: float = 0.25
    max_vr_speed_m_s: float = 2.0
    spike_recovery_frames: int = 2
    compute_anchor_shadow: bool = True
    motion_shaping_mode: str = "robot_controller"
    position_cutoff_hz: float = 5.0
    max_linear_speed_m_s: float = 0.25
    max_linear_accel_m_s2: float = 1.0
    engage_ramp_s: float = 0.35
    require_squeeze_to_move: bool = True
    gripper_thumbstick_axis: str = "x"
    gripper_thumbstick_deadzone: float = 0.05

    axis_remap: tuple[tuple[float, float, float], ...] = field(
        default_factory=lambda: AXIS_REMAP_VR_TO_ROBOT
    )

    _prev_position: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_orientation_quat: np.ndarray | None = field(default=None, init=False, repr=False)
    _control_was_active: bool = field(default=False, init=False, repr=False)
    _prev_sample_receive_time_s: float | None = field(default=None, init=False, repr=False)
    _last_observed_sequence: int | None = field(default=None, init=False, repr=False)
    _command_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float), init=False, repr=False
    )
    _active_elapsed_s: float = field(default=0.0, init=False, repr=False)
    _control_state: str = field(default="IDLE", init=False, repr=False)
    _spike_candidate_position: np.ndarray | None = field(default=None, init=False, repr=False)
    _spike_candidate_time_s: float | None = field(default=None, init=False, repr=False)
    _spike_count: int = field(default=0, init=False, repr=False)
    _position_filter: _SecondOrderLowPass = field(init=False, repr=False)
    _diagnostics_frame_index: int = field(default=0, init=False, repr=False)
    _current_diagnostics: dict = field(default_factory=dict, init=False, repr=False)
    _last_diagnostics: dict | None = field(default=None, init=False, repr=False)
    _anchor_shadow_session_id: str = field(
        default_factory=lambda: uuid.uuid4().hex, init=False, repr=False
    )
    _anchor_shadow_id: int = field(default=0, init=False, repr=False)
    _anchor_shadow_active: bool = field(default=False, init=False, repr=False)
    _anchor_shadow_position_vr_m: np.ndarray | None = field(default=None, init=False, repr=False)
    _anchor_shadow_quat: np.ndarray | None = field(default=None, init=False, repr=False)
    _anchor_shadow_sample_sequence: int | None = field(default=None, init=False, repr=False)
    _anchor_shadow_receive_time_s: float | None = field(default=None, init=False, repr=False)
    _anchor_shadow_pos_scale: float = field(default=0.0, init=False, repr=False)
    _anchor_shadow_angle_scale: float = field(default=0.0, init=False, repr=False)
    _anchor_shadow_last_translation_ee_m: np.ndarray | None = field(
        default=None, init=False, repr=False
    )
    _anchor_shadow_last_rotation_ee_rad: np.ndarray | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.position_control_mode not in {"safe_frame_delta", "legacy_frame_delta"}:
            raise ValueError(
                "position_control_mode must be 'safe_frame_delta' or 'legacy_frame_delta'"
            )
        if self.motion_shaping_mode not in {"robot_controller", "lerobot_a1"}:
            raise ValueError(
                "motion_shaping_mode must be 'robot_controller' or 'lerobot_a1'"
            )
        finite_values = {
            "pos_scale": self.pos_scale,
            "angle_scale": self.angle_scale,
            "vr_to_robot_scale": self.vr_to_robot_scale,
            "fine_trigger_threshold": self.fine_trigger_threshold,
            "fine_scale_factor": self.fine_scale_factor,
            "pos_deadzone_m": self.pos_deadzone_m,
            "angle_deadzone_deg": self.angle_deadzone_deg,
            "stale_timeout_s": self.stale_timeout_s,
            "max_vr_speed_m_s": self.max_vr_speed_m_s,
            "position_cutoff_hz": self.position_cutoff_hz,
            "max_linear_speed_m_s": self.max_linear_speed_m_s,
            "max_linear_accel_m_s2": self.max_linear_accel_m_s2,
            "engage_ramp_s": self.engage_ramp_s,
            "gripper_thumbstick_deadzone": self.gripper_thumbstick_deadzone,
        }
        nonfinite = [name for name, value in finite_values.items() if not math.isfinite(value)]
        if nonfinite:
            raise ValueError(f"non-finite XLeVR config values: {', '.join(nonfinite)}")
        if self.stale_timeout_s <= 0.0:
            raise ValueError("stale_timeout_s must be positive")
        if self.max_vr_speed_m_s <= 0.0:
            raise ValueError("max_vr_speed_m_s must be positive")
        if self.spike_recovery_frames < 1:
            raise ValueError("spike_recovery_frames must be at least 1")
        if self.position_cutoff_hz < 0.0:
            raise ValueError("position_cutoff_hz cannot be negative")
        if self.max_linear_speed_m_s <= 0.0:
            raise ValueError("max_linear_speed_m_s must be positive")
        if self.max_linear_accel_m_s2 <= 0.0:
            raise ValueError("max_linear_accel_m_s2 must be positive")
        if self.engage_ramp_s < 0.0:
            raise ValueError("engage_ramp_s cannot be negative")
        if self.max_delta_pos_m is not None and (
            not math.isfinite(self.max_delta_pos_m) or self.max_delta_pos_m <= 0.0
        ):
            raise ValueError("max_delta_pos_m must be positive when configured")
        if self.max_delta_angle_deg is not None and (
            not math.isfinite(self.max_delta_angle_deg) or self.max_delta_angle_deg <= 0.0
        ):
            raise ValueError("max_delta_angle_deg must be positive when configured")
        if (
            self.pos_deadzone_m < 0.0
            or self.angle_deadzone_deg < 0.0
            or self.gripper_thumbstick_deadzone < 0.0
        ):
            raise ValueError("deadzone values cannot be negative")
        self._position_filter = _SecondOrderLowPass(self.position_cutoff_hz)

    @property
    def control_state(self) -> str:
        return self._control_state

    @staticmethod
    def _diagnostic_number(value) -> float | int | None:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None

    @classmethod
    def _diagnostic_vector(cls, value, size: int) -> list[float | None] | None:
        try:
            vector = np.asarray(value, dtype=float).reshape(size)
        except (TypeError, ValueError):
            return None
        return [cls._diagnostic_number(item) for item in vector]

    def _begin_diagnostics(self, action: RobotAction) -> None:
        thumbstick = action.get("xlevr.thumbstick", {}) or {}
        buttons = action.get("xlevr.buttons", {}) or {}
        if not isinstance(thumbstick, dict):
            thumbstick = {}
        if not isinstance(buttons, dict):
            buttons = {}
        self._current_diagnostics = {
            "schema_version": 1,
            "algorithm": "xlevr_a1_position_control",
            "anchor_shadow_enabled": self.compute_anchor_shadow,
            "anchor_shadow_session_id": (
                self._anchor_shadow_session_id if self.compute_anchor_shadow else None
            ),
            "anchor_shadow_active": self._anchor_shadow_active,
            "anchor_shadow_status": "inactive",
            "position_control_mode": self.position_control_mode,
            "motion_shaping_mode": self.motion_shaping_mode,
            "motion_shaping_owner": (
                "a10_vr_vel"
                if self.motion_shaping_mode == "robot_controller"
                else "lerobot_a1"
            ),
            "mapper_frame_index": self._diagnostics_frame_index,
            "sample_sequence": self._diagnostic_number(
                action.get("xlevr.sample_sequence")
            ),
            "sample_receive_time_s": self._diagnostic_number(
                action.get("xlevr.sample_receive_time_s")
            ),
            "sample_age_s": self._diagnostic_number(action.get("xlevr.sample_age_s")),
            "source_timestamp": self._diagnostic_number(
                action.get("xlevr.source_timestamp")
            ),
            "raw_position_vr_m": self._diagnostic_vector(
                action.get("xlevr.target_position"), 3
            ),
            "raw_orientation_quat_xyzw": self._diagnostic_vector(
                action.get("xlevr.orientation_quat"), 4
            ),
            "raw_enabled": bool(action.get("xlevr.enabled", False)),
            "raw_grip_active": bool(action.get("xlevr.grip_active", False)),
            "raw_trigger": self._diagnostic_number(action.get("xlevr.trigger", 0.0)),
            "raw_thumbstick": {
                "x": self._diagnostic_number(thumbstick.get("x", 0.0)),
                "y": self._diagnostic_number(thumbstick.get("y", 0.0)),
            },
            "raw_buttons": {str(key): bool(value) for key, value in buttons.items()},
            "guard_reason": "unclassified",
        }
        self._diagnostics_frame_index += 1

    def _mark_diagnostics(self, reason: str, **values) -> None:
        self._current_diagnostics["guard_reason"] = reason
        self._current_diagnostics.update(values)

    def _finish_diagnostics(self, output: RobotAction) -> None:
        self._current_diagnostics.update(
            {
                "control_state": self._control_state,
                "ee_enabled": bool(output.get("ee.enabled", False)),
                "final_position_delta_m": [
                    float(output.get(f"ee.delta_{axis}", 0.0)) for axis in "xyz"
                ],
                "final_rotation_delta_rad": [
                    float(output.get(f"ee.delta_r{axis}", 0.0)) for axis in "xyz"
                ],
                "gripper_command": float(output.get("gripper.pos", 0.0)),
            }
        )
        self._last_diagnostics = copy.deepcopy(self._current_diagnostics)

    def get_last_diagnostics(self) -> dict | None:
        return copy.deepcopy(self._last_diagnostics)

    @property
    def _angle_deadzone_rad(self) -> float:
        return math.radians(self.angle_deadzone_deg)

    @property
    def _max_delta_angle_rad(self) -> float | None:
        if self.max_delta_angle_deg is None:
            return None
        return math.radians(self.max_delta_angle_deg)

    def _reset_reference(self) -> None:
        self._prev_position = None
        self._prev_orientation_quat = None
        self._prev_sample_receive_time_s = None
        self._last_observed_sequence = None

    def _reset_motion_state(self) -> None:
        self._position_filter.reset()
        self._command_velocity.fill(0.0)
        self._active_elapsed_s = 0.0
        self._spike_candidate_position = None
        self._spike_candidate_time_s = None
        self._spike_count = 0

    def _deactivate(self, state: str = "IDLE") -> None:
        self._reset_reference()
        self._reset_motion_state()
        self._reset_anchor_shadow()
        self._control_was_active = False
        self._control_state = state
        if self._current_diagnostics:
            self._set_anchor_shadow_status(state.lower())

    def _anchor_sample(
        self, position: np.ndarray, quat: np.ndarray, receive_time_s: float
    ) -> None:
        self._prev_position = position.copy()
        self._prev_orientation_quat = quat.copy()
        self._prev_sample_receive_time_s = receive_time_s

    def reset(self) -> None:
        self._deactivate()
        self._last_diagnostics = None

    def _thumbstick_to_gripper(self, thumbstick: dict) -> float:
        raw = float(thumbstick.get(self.gripper_thumbstick_axis, 0.0))
        if abs(raw) < self.gripper_thumbstick_deadzone:
            return 0.0
        return raw

    def _fine_trigger_active(self, trigger: float) -> bool:
        return trigger >= self.fine_trigger_threshold

    def _motion_scales(self, trigger: float) -> tuple[float, float]:
        if self._fine_trigger_active(trigger):
            return (
                self.pos_scale * self.fine_scale_factor,
                self.angle_scale * self.fine_scale_factor,
            )
        return self.pos_scale, self.angle_scale

    def _reset_anchor_shadow(self) -> None:
        self._anchor_shadow_active = False
        self._anchor_shadow_position_vr_m = None
        self._anchor_shadow_quat = None
        self._anchor_shadow_sample_sequence = None
        self._anchor_shadow_receive_time_s = None
        self._anchor_shadow_pos_scale = 0.0
        self._anchor_shadow_angle_scale = 0.0
        self._anchor_shadow_last_translation_ee_m = None
        self._anchor_shadow_last_rotation_ee_rad = None

    def _set_anchor_shadow_status(self, status: str) -> None:
        self._current_diagnostics.update(
            {
                "anchor_shadow_active": self._anchor_shadow_active,
                "anchor_shadow_status": status,
                "anchor_shadow_transmitted": False,
            }
        )

    def _start_anchor_shadow(
        self,
        position_vr_m: np.ndarray,
        quat: np.ndarray,
        trigger: float,
        sample_sequence: int,
        receive_time_s: float,
        *,
        status: str = "anchored",
    ) -> None:
        if not self.compute_anchor_shadow:
            self._set_anchor_shadow_status("disabled")
            return
        pos_scale, angle_scale = self._motion_scales(trigger)
        self._anchor_shadow_id += 1
        self._anchor_shadow_active = True
        self._anchor_shadow_position_vr_m = position_vr_m.copy()
        self._anchor_shadow_quat = quat.copy()
        self._anchor_shadow_sample_sequence = sample_sequence
        self._anchor_shadow_receive_time_s = receive_time_s
        self._anchor_shadow_pos_scale = pos_scale
        self._anchor_shadow_angle_scale = angle_scale
        self._record_anchor_shadow(position_vr_m, quat, status=status)

    def _record_anchor_shadow(
        self, position_vr_m: np.ndarray, quat: np.ndarray, *, status: str
    ) -> None:
        if not self.compute_anchor_shadow:
            self._set_anchor_shadow_status("disabled")
            return
        if (
            not self._anchor_shadow_active
            or self._anchor_shadow_position_vr_m is None
            or self._anchor_shadow_quat is None
        ):
            self._set_anchor_shadow_status(status)
            return

        if (
            status == "duplicate"
            and self._anchor_shadow_last_translation_ee_m is not None
            and self._anchor_shadow_last_rotation_ee_rad is not None
        ):
            translation_final = self._anchor_shadow_last_translation_ee_m.copy()
            rotation_final = self._anchor_shadow_last_rotation_ee_rad.copy()
        else:
            displacement_anchor_vr = delta_position_body(
                self._anchor_shadow_position_vr_m,
                position_vr_m,
                self._anchor_shadow_quat,
            )
            translation_ee = remap_position(
                displacement_anchor_vr * self.vr_to_robot_scale * self._anchor_shadow_pos_scale,
                self.axis_remap,
            )
            rotation_anchor_vr = (
                quat_delta_rotvec_body_rad(self._anchor_shadow_quat, quat)
                * self._anchor_shadow_angle_scale
            )
            rotation_ee = remap_rotvec(rotation_anchor_vr, self.axis_remap)
            translation_final = self._apply_deadzone_pos(translation_ee)
            rotation_final = np.asarray(
                self._apply_deadzone_rotvec(*[float(value) for value in rotation_ee]),
                dtype=float,
            )
            self._anchor_shadow_last_translation_ee_m = translation_final.copy()
            self._anchor_shadow_last_rotation_ee_rad = rotation_final.copy()
        self._current_diagnostics.update(
            {
                "anchor_shadow_protocol": "SET_EE_ANCHOR/v1",
                "anchor_shadow_transmitted": False,
                "anchor_shadow_active": True,
                "anchor_shadow_status": status,
                "anchor_shadow_id": self._anchor_shadow_id,
                "anchor_shadow_anchor_sample_sequence": self._anchor_shadow_sample_sequence,
                "anchor_shadow_anchor_receive_time_s": self._anchor_shadow_receive_time_s,
                "anchor_shadow_position_vr_m": self._anchor_shadow_position_vr_m.tolist(),
                "anchor_shadow_quat_xyzw": self._anchor_shadow_quat.tolist(),
                "anchor_shadow_pos_scale": self._anchor_shadow_pos_scale,
                "anchor_shadow_angle_scale": self._anchor_shadow_angle_scale,
                "anchor_shadow_translation_ee_m": translation_final.tolist(),
                "anchor_shadow_rotation_ee_rad": rotation_final.tolist(),
                "anchor_shadow_offset_6d": np.concatenate(
                    (translation_final, rotation_final)
                ).tolist(),
            }
        )

    def _orientation_delta_rotvec(
        self,
        orientation_quat,
        *,
        angle_scale: float,
    ) -> tuple[float, float, float]:
        quat = parse_quat_xyzw(orientation_quat)
        if quat is None:
            return 0.0, 0.0, 0.0
        if self._prev_orientation_quat is None:
            self._prev_orientation_quat = quat.copy()
            return 0.0, 0.0, 0.0

        rotvec_vr = quat_delta_rotvec_body_rad(self._prev_orientation_quat, quat) * angle_scale
        self._prev_orientation_quat = quat.copy()
        rotvec_robot = remap_rotvec(rotvec_vr, self.axis_remap)
        rx, ry, rz = float(rotvec_robot[0]), float(rotvec_robot[1]), float(rotvec_robot[2])
        rx, ry, rz = self._clip_delta_rotvec(rx, ry, rz)
        rx, ry, rz = self._apply_deadzone_rotvec(rx, ry, rz)
        return rx, ry, rz

    def action(self, action: RobotAction) -> RobotAction:
        self._begin_diagnostics(action)
        if self.position_control_mode == "legacy_frame_delta":
            output = self._legacy_action(action)
        else:
            output = self._safe_action(action)
        self._finish_diagnostics(output)
        return output

    def _legacy_action(self, action: RobotAction) -> RobotAction:
        action.pop("xlevr.enabled", False)
        target_position = action.pop("xlevr.target_position", None)
        orientation_quat = action.pop("xlevr.orientation_quat", None)
        grip_active = bool(action.pop("xlevr.grip_active", False))
        trigger = float(action.pop("xlevr.trigger", 0.0))
        thumbstick = action.pop("xlevr.thumbstick", {}) or {}
        buttons = action.pop("xlevr.buttons", {}) or {}

        gripper = self._thumbstick_to_gripper(thumbstick)
        if grip_active:
            buttons = dict(buttons)
            buttons["squeeze"] = True
        squeeze_active = bool(buttons.get("squeeze", False))
        fine_active = self._fine_trigger_active(trigger)
        if self.require_squeeze_to_move:
            control_active = squeeze_active or fine_active
        else:
            control_active = True
        pos_scale_eff, angle_scale_eff = self._motion_scales(trigger)

        if not control_active:
            self._control_state = "IDLE"
            self._mark_diagnostics(
                "legacy_idle",
                control_requested=False,
            )
            if self._control_was_active:
                self._reset_reference()
            self._control_was_active = False
            return self._disabled_arm_action(gripper, trigger, thumbstick, buttons)

        if not self._control_was_active:
            self._reset_reference()
        self._control_was_active = True

        delta_pos = np.zeros(3, dtype=float)
        if target_position is not None:
            current_pos = np.asarray(target_position, dtype=float) * self.vr_to_robot_scale
            if self._prev_position is None:
                self._prev_position = current_pos.copy()
            else:
                if self._prev_orientation_quat is not None:
                    raw_delta = (
                        delta_position_body(
                            self._prev_position,
                            current_pos,
                            self._prev_orientation_quat,
                        )
                        * pos_scale_eff
                    )
                else:
                    raw_delta = (current_pos - self._prev_position) * pos_scale_eff
                self._current_diagnostics["mapped_delta_before_filter_m"] = raw_delta.tolist()
                self._prev_position = current_pos.copy()
                delta_pos = remap_position(raw_delta, self.axis_remap)
                delta_pos = self._clip_delta_pos(delta_pos)

        delta_rx, delta_ry, delta_rz = self._orientation_delta_rotvec(
            orientation_quat,
            angle_scale=angle_scale_eff,
        )
        delta_pos = self._apply_deadzone_pos(delta_pos)

        self._control_state = "ACTIVE"
        self._mark_diagnostics("legacy_active", control_requested=True)

        return {
            "ee.enabled": True,
            "ee.delta_x": float(delta_pos[0]),
            "ee.delta_y": float(delta_pos[1]),
            "ee.delta_z": float(delta_pos[2]),
            "ee.delta_rx": delta_rx,
            "ee.delta_ry": delta_ry,
            "ee.delta_rz": delta_rz,
            "gripper.pos": gripper,
        }

    def _safe_action(self, action: RobotAction) -> RobotAction:
        enabled = bool(action.pop("xlevr.enabled", False))
        target_position = action.pop("xlevr.target_position", None)
        orientation_quat = action.pop("xlevr.orientation_quat", None)
        grip_active = bool(action.pop("xlevr.grip_active", False))
        try:
            trigger = float(action.pop("xlevr.trigger", 0.0))
        except (TypeError, ValueError):
            trigger = float("nan")
        thumbstick = action.pop("xlevr.thumbstick", {}) or {}
        buttons = action.pop("xlevr.buttons", {}) or {}
        sample_receive_time_s = action.pop("xlevr.sample_receive_time_s", -1.0)
        sample_age_s = action.pop("xlevr.sample_age_s", -1.0)
        sample_sequence = action.pop("xlevr.sample_sequence", -1)
        action.pop("xlevr.source_timestamp", -1.0)

        try:
            sample_receive_time_s = float(sample_receive_time_s)
            sample_age_s = float(sample_age_s)
            sample_sequence = int(sample_sequence)
        except (TypeError, ValueError, OverflowError):
            sample_receive_time_s = -1.0
            sample_age_s = -1.0
            sample_sequence = -1

        try:
            gripper = self._thumbstick_to_gripper(thumbstick)
        except (AttributeError, TypeError, ValueError):
            gripper = 0.0
        if not math.isfinite(gripper):
            gripper = 0.0
        if grip_active:
            buttons = dict(buttons)
            buttons["squeeze"] = True

        self._current_diagnostics.update(
            {
                "control_requested": enabled,
                "effective_squeeze": bool(buttons.get("squeeze", False)),
                "gripper_command": gripper,
            }
        )
        if not enabled:
            self._deactivate()
            self._mark_diagnostics("idle")
            return self._disabled_arm_action(gripper, trigger, thumbstick, buttons)

        try:
            vr_position = np.asarray(target_position, dtype=float).reshape(3)
        except (TypeError, ValueError):
            vr_position = np.full(3, np.nan)
        quat = parse_quat_xyzw(orientation_quat)
        valid_sample = (
            np.all(np.isfinite(vr_position))
            and quat is not None
            and math.isfinite(trigger)
            and math.isfinite(sample_receive_time_s)
            and sample_receive_time_s >= 0.0
            and math.isfinite(sample_age_s)
            and sample_age_s >= 0.0
            and sample_sequence >= 0
        )
        if not valid_sample:
            self._deactivate("FAULT")
            self._mark_diagnostics("invalid_sample")
            return self._disabled_arm_action(gripper, trigger, thumbstick, buttons)
        if sample_age_s > self.stale_timeout_s:
            self._deactivate("STALE")
            self._mark_diagnostics("stale_age")
            return self._disabled_arm_action(gripper, trigger, thumbstick, buttons)

        current_pos = vr_position * self.vr_to_robot_scale
        if not self._control_was_active:
            self._reset_reference()
            self._reset_motion_state()
            self._control_was_active = True
            self._control_state = "ARMING"
            self._last_observed_sequence = sample_sequence
            self._anchor_sample(current_pos, quat, sample_receive_time_s)
            self._start_anchor_shadow(
                vr_position,
                quat,
                trigger,
                sample_sequence,
                sample_receive_time_s,
            )
            self._mark_diagnostics("arming")
            return self._disabled_arm_action(gripper, trigger, thumbstick, buttons)

        last_observed_sequence = self._last_observed_sequence
        if last_observed_sequence is None or sample_sequence < last_observed_sequence:
            self._deactivate("FAULT")
            self._mark_diagnostics("sequence_rollback")
            return self._disabled_arm_action(gripper, trigger, thumbstick, buttons)
        if sample_sequence == last_observed_sequence:
            self._record_anchor_shadow(vr_position, quat, status="duplicate")
            self._mark_diagnostics("duplicate_sample")
            return self._safe_output(
                self._control_state == "ACTIVE",
                np.zeros(3, dtype=float),
                (0.0, 0.0, 0.0),
                gripper,
                trigger,
                thumbstick,
                buttons,
            )
        self._last_observed_sequence = sample_sequence

        previous_time_s = self._prev_sample_receive_time_s
        previous_position = self._prev_position
        previous_quat = self._prev_orientation_quat
        if previous_time_s is None or previous_position is None or previous_quat is None:
            self._deactivate("FAULT")
            self._mark_diagnostics("missing_reference")
            return self._disabled_arm_action(gripper, trigger, thumbstick, buttons)

        sample_dt_s = sample_receive_time_s - previous_time_s
        if sample_dt_s <= 0.0 or sample_dt_s > self.stale_timeout_s:
            self._deactivate("STALE" if sample_dt_s > self.stale_timeout_s else "FAULT")
            self._mark_diagnostics(
                "sample_gap" if sample_dt_s > self.stale_timeout_s else "timestamp_regression",
                sample_dt_s=sample_dt_s,
            )
            return self._disabled_arm_action(gripper, trigger, thumbstick, buttons)

        scale_abs = max(abs(self.vr_to_robot_scale), 1e-12)
        vr_speed_m_s = np.linalg.norm(current_pos - previous_position) / scale_abs / sample_dt_s
        if vr_speed_m_s > self.max_vr_speed_m_s:
            recovered = self._update_spike_candidate(
                current_pos, sample_receive_time_s, scale_abs
            )
            self._position_filter.reset()
            self._command_velocity.fill(0.0)
            self._active_elapsed_s = 0.0
            self._reset_anchor_shadow()
            if recovered:
                self._anchor_sample(current_pos, quat, sample_receive_time_s)
                self._control_state = "ARMING"
                self._start_anchor_shadow(
                    vr_position,
                    quat,
                    trigger,
                    sample_sequence,
                    sample_receive_time_s,
                    status="reanchored",
                )
            else:
                self._control_state = "FAULT"
                self._set_anchor_shadow_status("spike_pending")
            self._mark_diagnostics(
                "spike_reanchor" if recovered else "spike_pending",
                sample_dt_s=sample_dt_s,
                vr_input_speed_m_s=float(vr_speed_m_s),
            )
            return self._disabled_arm_action(gripper, trigger, thumbstick, buttons)

        self._spike_candidate_position = None
        self._spike_candidate_time_s = None
        self._spike_count = 0
        pos_scale_eff, angle_scale_eff = self._motion_scales(trigger)
        raw_delta = delta_position_body(previous_position, current_pos, previous_quat) * pos_scale_eff
        delta_pos = remap_position(raw_delta, self.axis_remap)
        rotvec_vr = quat_delta_rotvec_body_rad(previous_quat, quat) * angle_scale_eff
        rotvec_robot = remap_rotvec(rotvec_vr, self.axis_remap)
        self._anchor_sample(current_pos, quat, sample_receive_time_s)
        self._record_anchor_shadow(vr_position, quat, status="active")

        unfiltered_velocity = delta_pos / sample_dt_s
        if self.motion_shaping_mode == "lerobot_a1":
            filtered_velocity = self._position_filter.update(
                unfiltered_velocity, sample_dt_s
            )
            speed_limited_velocity = self._limit_vector(
                filtered_velocity, self.max_linear_speed_m_s
            )
            speed_limited = not np.array_equal(
                speed_limited_velocity, filtered_velocity
            )
            self._active_elapsed_s += sample_dt_s
            ramp_gain = 1.0
            if self.engage_ramp_s > 0.0:
                ramp = min(1.0, self._active_elapsed_s / self.engage_ramp_s)
                ramp_gain = ramp * ramp * (3.0 - 2.0 * ramp)
            ramped_velocity = speed_limited_velocity * ramp_gain
            requested_velocity_change = ramped_velocity - self._command_velocity
            limited_velocity_change = self._limit_vector(
                requested_velocity_change, self.max_linear_accel_m_s2 * sample_dt_s
            )
            acceleration_limited = not np.array_equal(
                limited_velocity_change, requested_velocity_change
            )
            self._command_velocity += limited_velocity_change
            delta_pos = self._command_velocity * sample_dt_s
        else:
            # The validated A10 ``vr_vel`` loop owns low-pass/speed/acceleration
            # shaping at 500 Hz. Keep LeRobot as a guarded coordinate mapper so
            # the command is shaped exactly once.
            filtered_velocity = unfiltered_velocity.copy()
            speed_limited_velocity = unfiltered_velocity.copy()
            speed_limited = False
            acceleration_limited = False
            ramp_gain = 1.0
            self._command_velocity = unfiltered_velocity.copy()
        delta_capped = False
        if self.max_delta_pos_m is not None:
            capped_delta_pos = self._limit_vector(delta_pos, self.max_delta_pos_m)
            delta_capped = not np.array_equal(capped_delta_pos, delta_pos)
            delta_pos = capped_delta_pos
        delta_pos = self._apply_deadzone_pos(delta_pos)

        delta_rx, delta_ry, delta_rz = self._clip_delta_rotvec(
            float(rotvec_robot[0]), float(rotvec_robot[1]), float(rotvec_robot[2])
        )
        delta_rx, delta_ry, delta_rz = self._apply_deadzone_rotvec(
            delta_rx, delta_ry, delta_rz
        )
        self._control_state = "ACTIVE"
        self._mark_diagnostics(
            "active",
            sample_dt_s=sample_dt_s,
            vr_input_speed_m_s=float(vr_speed_m_s),
            body_delta_before_remap_m=raw_delta.tolist(),
            mapped_delta_before_filter_m=(unfiltered_velocity * sample_dt_s).tolist(),
            unfiltered_velocity_m_s=unfiltered_velocity.tolist(),
            filtered_velocity_m_s=filtered_velocity.tolist(),
            speed_limited_velocity_m_s=speed_limited_velocity.tolist(),
            speed_limited=speed_limited,
            ramp_gain=ramp_gain,
            acceleration_limited=acceleration_limited,
            shaping_bypassed=self.motion_shaping_mode == "robot_controller",
            commanded_velocity_m_s=self._command_velocity.tolist(),
            frame_delta_limited=delta_capped,
        )
        return self._safe_output(
            True,
            delta_pos,
            (delta_rx, delta_ry, delta_rz),
            gripper,
            trigger,
            thumbstick,
            buttons,
        )

    def _update_spike_candidate(
        self, position: np.ndarray, receive_time_s: float, vr_scale_abs: float
    ) -> bool:
        if self._spike_candidate_position is None or self._spike_candidate_time_s is None:
            self._spike_count = 1
        else:
            dt_s = receive_time_s - self._spike_candidate_time_s
            candidate_speed = math.inf
            if dt_s > 0.0:
                candidate_speed = (
                    np.linalg.norm(position - self._spike_candidate_position)
                    / vr_scale_abs
                    / dt_s
                )
            if candidate_speed <= self.max_vr_speed_m_s:
                self._spike_count += 1
            else:
                self._spike_count = 1
        self._spike_candidate_position = position.copy()
        self._spike_candidate_time_s = receive_time_s
        if self._spike_count < self.spike_recovery_frames:
            return False
        self._spike_candidate_position = None
        self._spike_candidate_time_s = None
        self._spike_count = 0
        return True

    @staticmethod
    def _limit_vector(value: np.ndarray, max_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(value))
        if norm <= max_norm or norm < 1e-12:
            return value
        return value * (max_norm / norm)

    def _safe_output(
        self,
        enabled: bool,
        delta_pos: np.ndarray,
        delta_rotvec: tuple[float, float, float],
        gripper: float,
        trigger: float,
        thumbstick: dict,
        buttons: dict,
    ) -> RobotAction:
        output = self._disabled_arm_action(gripper, trigger, thumbstick, buttons)
        output.update(
            {
                "ee.enabled": enabled,
                "ee.delta_x": float(delta_pos[0]),
                "ee.delta_y": float(delta_pos[1]),
                "ee.delta_z": float(delta_pos[2]),
                "ee.delta_rx": delta_rotvec[0],
                "ee.delta_ry": delta_rotvec[1],
                "ee.delta_rz": delta_rotvec[2],
            }
        )
        return output

    def _disabled_arm_action(
        self,
        gripper: float,
        trigger: float,
        thumbstick: dict,
        buttons: dict,
    ) -> RobotAction:
        return {
            "ee.enabled": False,
            "ee.delta_x": 0.0,
            "ee.delta_y": 0.0,
            "ee.delta_z": 0.0,
            "ee.delta_rx": 0.0,
            "ee.delta_ry": 0.0,
            "ee.delta_rz": 0.0,
            "gripper.pos": gripper,
        }

    def _apply_deadzone_pos(self, delta: np.ndarray) -> np.ndarray:
        out = delta.copy()
        for i in range(3):
            if abs(out[i]) < self.pos_deadzone_m:
                out[i] = 0.0
        return out

    def _apply_deadzone_rotvec(self, rx: float, ry: float, rz: float) -> tuple[float, float, float]:
        dz = self._angle_deadzone_rad
        if abs(rx) < dz:
            rx = 0.0
        if abs(ry) < dz:
            ry = 0.0
        if abs(rz) < dz:
            rz = 0.0
        return rx, ry, rz

    def _clip_delta_pos(self, delta: np.ndarray) -> np.ndarray:
        if self.max_delta_pos_m is None:
            return delta
        return np.clip(delta, -self.max_delta_pos_m, self.max_delta_pos_m)

    def _clip_delta_rotvec(self, rx: float, ry: float, rz: float) -> tuple[float, float, float]:
        cap = self._max_delta_angle_rad
        if cap is None:
            return rx, ry, rz
        return (
            float(np.clip(rx, -cap, cap)),
            float(np.clip(ry, -cap, cap)),
            float(np.clip(rz, -cap, cap)),
        )

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in (
            "enabled",
            "target_position",
            "orientation_quat",
            "grip_active",
            "trigger",
            "thumbstick",
            "buttons",
            "sample_receive_time_s",
            "sample_age_s",
            "sample_sequence",
            "source_timestamp",
        ):
            features[PipelineFeatureType.ACTION].pop(f"xlevr.{feat}", None)

        ee_features = {
            "ee.enabled": FeatureType.STATE,
            "ee.delta_x": FeatureType.STATE,
            "ee.delta_y": FeatureType.STATE,
            "ee.delta_z": FeatureType.STATE,
            "ee.delta_rx": FeatureType.STATE,
            "ee.delta_ry": FeatureType.STATE,
            "ee.delta_rz": FeatureType.STATE,
            "gripper.pos": FeatureType.STATE,
        }
        for key, feat_type in ee_features.items():
            features[PipelineFeatureType.ACTION][key] = PolicyFeature(type=feat_type, shape=(1,))
        return features
