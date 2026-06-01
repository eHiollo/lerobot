#!/usr/bin/env python

from dataclasses import dataclass, field

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotAction, RobotActionProcessorStep


@ProcessorStepRegistry.register("xlevr_to_ee_delta")
@dataclass
class XLeVRDeltaEEMapper(RobotActionProcessorStep):
    """
    Convert consecutive XLeVR absolute poses into end-effector delta commands.

    Squeeze button gates arm motion; thumbstick x is passed through as gripper.pos.
    No inverse kinematics here — the robot controller is expected to consume
    `ee.delta_*` and run IK locally.
    """

    pos_scale: float = 1.0
    angle_scale: float = 1.0
    vr_to_robot_scale: float = 1.0
    pos_deadzone_m: float = 0.0005
    angle_deadzone_deg: float = 0.05
    max_delta_pos_m: float | None = None
    max_delta_angle_deg: float | None = None
    require_squeeze_to_move: bool = True
    gripper_thumbstick_axis: str = "x"
    gripper_thumbstick_deadzone: float = 0.05

    # VR (+X right, +Y up, +Z back) -> robot (+X fwd, +Y left, +Z up)
    axis_remap: tuple[tuple[float, float, float], ...] = field(
        default_factory=lambda: (
            (0.0, 0.0, -1.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    delta_roll_sign: float = -1.0
    delta_pitch_sign: float = 1.0
    delta_yaw_sign: float = -1.0

    _prev_position: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_wrist_roll: float | None = field(default=None, init=False, repr=False)
    _prev_wrist_flex: float | None = field(default=None, init=False, repr=False)
    _prev_wrist_yaw: float | None = field(default=None, init=False, repr=False)
    _control_was_active: bool = field(default=False, init=False, repr=False)

    def _reset_reference(self) -> None:
        self._prev_position = None
        self._prev_wrist_roll = None
        self._prev_wrist_flex = None
        self._prev_wrist_yaw = None

    def _thumbstick_to_gripper(self, thumbstick: dict) -> float:
        raw = float(thumbstick.get(self.gripper_thumbstick_axis, 0.0))
        if abs(raw) < self.gripper_thumbstick_deadzone:
            return 0.0
        return raw

    def action(self, action: RobotAction) -> RobotAction:
        action.pop("xlevr.enabled", False)
        target_position = action.pop("xlevr.target_position", None)
        wrist_roll_deg = action.pop("xlevr.wrist_roll_deg", None)
        wrist_flex_deg = action.pop("xlevr.wrist_flex_deg", None)
        wrist_yaw_deg = action.pop("xlevr.wrist_yaw_deg", None)
        action.pop("xlevr.gripper_closed", None)
        grip_active = bool(action.pop("xlevr.grip_active", False))
        trigger = float(action.pop("xlevr.trigger", 0.0))
        thumbstick = action.pop("xlevr.thumbstick", {}) or {}
        buttons = action.pop("xlevr.buttons", {}) or {}

        gripper = self._thumbstick_to_gripper(thumbstick)
        if grip_active:
            buttons = dict(buttons)
            buttons["squeeze"] = True
        squeeze_active = bool(buttons.get("squeeze", False))
        control_active = squeeze_active if self.require_squeeze_to_move else True

        if not control_active:
            if self._control_was_active:
                self._reset_reference()
            self._control_was_active = False
            return self._disabled_arm_action(gripper, trigger, thumbstick, buttons)

        if not self._control_was_active:
            self._reset_reference()
        self._control_was_active = True

        delta_pos = np.zeros(3, dtype=float)
        delta_roll = 0.0
        delta_pitch = 0.0
        delta_yaw = 0.0

        if target_position is not None:
            current_pos = np.asarray(target_position, dtype=float) * self.vr_to_robot_scale
            if self._prev_position is None:
                self._prev_position = current_pos.copy()
            else:
                raw_delta = (current_pos - self._prev_position) * self.pos_scale
                self._prev_position = current_pos.copy()
                delta_pos = self._remap_position(raw_delta)
                delta_pos = self._clip_delta_pos(delta_pos)

            if wrist_roll_deg is not None:
                if self._prev_wrist_roll is None:
                    self._prev_wrist_roll = float(wrist_roll_deg)
                else:
                    delta_roll = (float(wrist_roll_deg) - self._prev_wrist_roll) * self.angle_scale
                    self._prev_wrist_roll = float(wrist_roll_deg)
                    delta_roll = self._clip_delta_angle(delta_roll)

            if wrist_flex_deg is not None:
                if self._prev_wrist_flex is None:
                    self._prev_wrist_flex = float(wrist_flex_deg)
                else:
                    delta_pitch = (float(wrist_flex_deg) - self._prev_wrist_flex) * self.angle_scale
                    self._prev_wrist_flex = float(wrist_flex_deg)
                    delta_pitch = self._clip_delta_angle(delta_pitch)

            if wrist_yaw_deg is not None:
                if self._prev_wrist_yaw is None:
                    self._prev_wrist_yaw = float(wrist_yaw_deg)
                else:
                    delta_yaw = (float(wrist_yaw_deg) - self._prev_wrist_yaw) * self.angle_scale
                    self._prev_wrist_yaw = float(wrist_yaw_deg)
                    delta_yaw = self._clip_delta_angle(delta_yaw)

        delta_roll, delta_pitch, delta_yaw = self._remap_angles(delta_roll, delta_pitch, delta_yaw)
        delta_pos = self._apply_deadzone_pos(delta_pos)
        delta_roll = self._apply_deadzone_angle(delta_roll)
        delta_pitch = self._apply_deadzone_angle(delta_pitch)
        delta_yaw = self._apply_deadzone_angle(delta_yaw)

        return {
            "ee.enabled": True,
            "ee.delta_x": float(delta_pos[0]),
            "ee.delta_y": float(delta_pos[1]),
            "ee.delta_z": float(delta_pos[2]),
            "ee.delta_roll": float(delta_roll),
            "ee.delta_pitch": float(delta_pitch),
            "ee.delta_yaw": float(delta_yaw),
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
            "ee.delta_roll": 0.0,
            "ee.delta_pitch": 0.0,
            "ee.delta_yaw": 0.0,
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

    def _remap_position(self, delta: np.ndarray) -> np.ndarray:
        matrix = np.asarray(self.axis_remap, dtype=float)
        return matrix @ delta

    def _remap_angles(self, roll: float, pitch: float, yaw: float) -> tuple[float, float, float]:
        # roll/pitch/yaw deltas are extracted about VR Z/X/Y respectively
        return (
            float(roll * self.delta_roll_sign),
            float(pitch * self.delta_pitch_sign),
            float(yaw * self.delta_yaw_sign),
        )

    def _apply_deadzone_pos(self, delta: np.ndarray) -> np.ndarray:
        out = delta.copy()
        for i in range(3):
            if abs(out[i]) < self.pos_deadzone_m:
                out[i] = 0.0
        return out

    def _apply_deadzone_angle(self, value: float) -> float:
        return 0.0 if abs(value) < self.angle_deadzone_deg else value

    def _clip_delta_pos(self, delta: np.ndarray) -> np.ndarray:
        if self.max_delta_pos_m is None:
            return delta
        return np.clip(delta, -self.max_delta_pos_m, self.max_delta_pos_m)

    def _clip_delta_angle(self, value: float) -> float:
        if self.max_delta_angle_deg is None:
            return value
        return float(np.clip(value, -self.max_delta_angle_deg, self.max_delta_angle_deg))

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in (
            "enabled",
            "target_position",
            "wrist_roll_deg",
            "wrist_flex_deg",
            "wrist_yaw_deg",
            "gripper_closed",
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
            "ee.delta_roll": FeatureType.STATE,
            "ee.delta_pitch": FeatureType.STATE,
            "ee.delta_yaw": FeatureType.STATE,
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
