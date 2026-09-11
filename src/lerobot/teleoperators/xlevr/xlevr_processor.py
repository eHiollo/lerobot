#!/usr/bin/env python

import math
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


@ProcessorStepRegistry.register("xlevr_to_ee_delta")
@dataclass
class XLeVRDeltaEEMapper(RobotActionProcessorStep):
    """
    Convert consecutive XLeVR poses into body-frame EE deltas, then remap to robot base.

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
    require_squeeze_to_move: bool = True
    gripper_thumbstick_axis: str = "x"
    gripper_thumbstick_deadzone: float = 0.05

    axis_remap: tuple[tuple[float, float, float], ...] = field(
        default_factory=lambda: AXIS_REMAP_VR_TO_ROBOT
    )

    _prev_position: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_orientation_quat: np.ndarray | None = field(default=None, init=False, repr=False)
    _control_was_active: bool = field(default=False, init=False, repr=False)

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
                self._prev_position = current_pos.copy()
                delta_pos = remap_position(raw_delta, self.axis_remap)
                delta_pos = self._clip_delta_pos(delta_pos)

        delta_rx, delta_ry, delta_rz = self._orientation_delta_rotvec(
            orientation_quat,
            angle_scale=angle_scale_eff,
        )
        delta_pos = self._apply_deadzone_pos(delta_pos)

        return {
            "ee.enabled": True,
            "ee.delta_x": float(delta_pos[0]),
            "ee.delta_y": float(delta_pos[1]),
            "ee.delta_z": float(delta_pos[2]),
            "ee.delta_rx": delta_rx,
            "ee.delta_ry": delta_ry,
            "ee.delta_rz": delta_rz,
            "gripper.pos": gripper,
            "vr.trigger": trigger,
            "vr.thumbstick_x": float(thumbstick.get("x", 0.0)),
            "vr.thumbstick_y": float(thumbstick.get("y", 0.0)),
            "vr.button_a": bool(buttons.get("a", False)),
            "vr.button_b": bool(buttons.get("b", False)),
            "vr.button_squeeze": squeeze_active,
            "vr.button_menu": bool(buttons.get("menu", False)),
            "vr.button_thumbstick": bool(buttons.get("thumbstick", False)),
        }

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
            "vr.trigger": trigger,
            "vr.thumbstick_x": float(thumbstick.get("x", 0.0)),
            "vr.thumbstick_y": float(thumbstick.get("y", 0.0)),
            "vr.button_a": bool(buttons.get("a", False)),
            "vr.button_b": bool(buttons.get("b", False)),
            "vr.button_squeeze": bool(buttons.get("squeeze", False)),
            "vr.button_menu": bool(buttons.get("menu", False)),
            "vr.button_thumbstick": bool(buttons.get("thumbstick", False)),
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
            "vr.trigger": FeatureType.STATE,
            "vr.thumbstick_x": FeatureType.STATE,
            "vr.thumbstick_y": FeatureType.STATE,
            "vr.button_a": FeatureType.STATE,
            "vr.button_b": FeatureType.STATE,
            "vr.button_squeeze": FeatureType.STATE,
            "vr.button_menu": FeatureType.STATE,
            "vr.button_thumbstick": FeatureType.STATE,
        }
        for key, feat_type in ee_features.items():
            features[PipelineFeatureType.ACTION][key] = PolicyFeature(type=feat_type, shape=(1,))
        return features
