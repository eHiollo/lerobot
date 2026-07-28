#!/usr/bin/env python

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotAction, RobotActionProcessorStep
from lerobot.teleoperators.xlevr.quaternion_utils import (
    delta_position_body,
    parse_quat_xyzw,
    quat_delta_rotvec_body_rad,
    remap_position,
    remap_rotvec,
)


class _OneEuroFilter1D:
    """精简版 One Euro 滤波器(单通道)：低频强平滑、高频低延迟。"""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, x: float, t: float | None = None) -> float:
        if self._t_prev is None or t is None:
            self._t_prev = t if t is not None else 0.0
            self._x_prev = x
            return x
        dt = max(t - self._t_prev, 1e-6)
        self._t_prev = t
        a_d = self._alpha(self.d_cutoff, dt)
        dx = a_d * (x - self._x_prev) / dt + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx)
        a = self._alpha(cutoff, dt)
        x_f = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_f
        self._dx_prev = dx
        return x_f


@dataclass
class _OneEuroFilterVec:
    """对 3D 向量逐通道做 One Euro 滤波。"""

    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0
    _fx: _OneEuroFilter1D = field(init=False, default_factory=lambda: _OneEuroFilter1D())
    _fy: _OneEuroFilter1D = field(init=False, default_factory=lambda: _OneEuroFilter1D())
    _fz: _OneEuroFilter1D = field(init=False, default_factory=lambda: _OneEuroFilter1D())

    def filter(self, v: np.ndarray, t: float | None = None) -> np.ndarray:
        return np.array(
            [
                self._fx.filter(float(v[0]), t),
                self._fy.filter(float(v[1]), t),
                self._fz.filter(float(v[2]), t),
            ],
            dtype=float,
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
        default_factory=lambda: (
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
        )
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
            elif self._prev_orientation_quat is not None:
                raw_delta = (
                    delta_position_body(
                        self._prev_position,
                        current_pos,
                        self._prev_orientation_quat,
                    )
                    * pos_scale_eff
                )
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


@ProcessorStepRegistry.register("xlevr_to_ee_target")
@dataclass
class XLeVRTargetEEMapper(RobotActionProcessorStep):
    """
    闭环"原点增量"映射器：按下 squeeze 时抓取机器人当前 EE 位姿作为 robot_origin、
    VR 当前位姿作为 vr_origin；之后每帧输出绝对末端目标

        target = robot_origin ∘ remap( vr_curr ⊖ vr_origin )

    通过 SET_EE_TARGET 直接替换机器人内部 target_pm(不累加)，VR 静止时 target 不变，
    SE3Follower 收敛后机器人停住，从根本上消除帧间增量累加噪声导致的爬行漂移。

    输入 action(来自 teleop.get_action()):
        xlevr.target_position / xlevr.orientation_quat / xlevr.trigger /
        xlevr.buttons(squeeze) / xlevr.thumbstick / xlevr.grip_active
    输入 obs(来自 robot.get_observation(), use_ee_target 模式):
        ee.x/y/z/rx/ry/rz  (机器人基坐标系, m / rad)
    输出:
        ee.enabled / ee.target_x/y/z/rx/ry/rz / gripper.pos / vr.*
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

    # One Euro 滤波(VR 位姿降噪)；min_cutoff 越小越平滑，beta 越大越跟手。
    enable_filter: bool = True
    filter_min_cutoff: float = 1.0
    filter_beta: float = 0.007

    axis_remap: tuple[tuple[float, float, float], ...] = field(
        default_factory=lambda: (
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
        )
    )

    _robot_origin_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _robot_origin_rot: R | None = field(default=None, init=False, repr=False)
    _vr_origin_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _vr_origin_quat: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_target_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_target_rotvec: np.ndarray | None = field(default=None, init=False, repr=False)
    _control_was_active: bool = field(default=False, init=False, repr=False)

    _pos_filter: _OneEuroFilterVec = field(init=False, repr=False)
    _quat_filter: _OneEuroFilterVec = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._pos_filter = _OneEuroFilterVec(self.filter_min_cutoff, self.filter_beta)
        self._quat_filter = _OneEuroFilterVec(self.filter_min_cutoff, self.filter_beta)

    @property
    def _angle_deadzone_rad(self) -> float:
        return math.radians(self.angle_deadzone_deg)

    @property
    def _max_delta_angle_rad(self) -> float | None:
        return None if self.max_delta_angle_deg is None else math.radians(self.max_delta_angle_deg)

    def _thumbstick_to_gripper(self, thumbstick: dict) -> float:
        raw = float(thumbstick.get(self.gripper_thumbstick_axis, 0.0))
        if abs(raw) < self.gripper_thumbstick_deadzone:
            return 0.0
        return raw

    def _fine_trigger_active(self, trigger: float) -> bool:
        return trigger >= self.fine_trigger_threshold

    def _motion_scales(self, trigger: float) -> tuple[float, float]:
        if self._fine_trigger_active(trigger):
            return (self.pos_scale * self.fine_scale_factor, self.angle_scale * self.fine_scale_factor)
        return (self.pos_scale, self.angle_scale)

    def _reset_origin(self) -> None:
        self._robot_origin_pos = None
        self._robot_origin_rot = None
        self._vr_origin_pos = None
        self._vr_origin_quat = None
        self._last_target_pos = None
        self._last_target_rotvec = None

    def _ee_pose_from_obs(self, obs: dict) -> tuple[np.ndarray, R] | None:
        try:
            pos = np.array([obs["ee.x"], obs["ee.y"], obs["ee.z"]], dtype=float)
            rotvec = np.array([obs["ee.rx"], obs["ee.ry"], obs["ee.rz"]], dtype=float)
        except KeyError:
            return None
        return pos, R.from_rotvec(rotvec)

    def _apply_deadzone_pos(self, delta: np.ndarray) -> np.ndarray:
        out = delta.copy()
        for i in range(3):
            if abs(out[i]) < self.pos_deadzone_m:
                out[i] = 0.0
        return out

    def _apply_deadzone_rotvec(self, rv: np.ndarray) -> np.ndarray:
        dz = self._angle_deadzone_rad
        out = rv.copy()
        for i in range(3):
            if abs(out[i]) < dz:
                out[i] = 0.0
        return out

    def _clip_pos(self, delta: np.ndarray) -> np.ndarray:
        if self.max_delta_pos_m is None:
            return delta
        return np.clip(delta, -self.max_delta_pos_m, self.max_delta_pos_m)

    def _clip_rotvec(self, rv: np.ndarray) -> np.ndarray:
        cap = self._max_delta_angle_rad
        if cap is None:
            return rv
        return np.clip(rv, -cap, cap)

    def action(self, action: RobotAction, obs: RobotAction) -> RobotAction:
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
        control_active = (squeeze_active or fine_active) if self.require_squeeze_to_move else True
        pos_scale, angle_scale = self._motion_scales(trigger)

        # 未激活：保持上一次目标(机器人 hold)，夹爪仍随摇杆更新。
        if not control_active:
            if self._control_was_active:
                self._reset_origin()
            self._control_was_active = False
            return self._hold_action(gripper, trigger, thumbstick, buttons)

        # 激活上升沿：抓取 robot_origin(来自 obs EE)与 vr_origin。
        if not self._control_was_active:
            self._reset_origin()
            ee = self._ee_pose_from_obs(obs)
            if ee is None:
                # 没有 EE 反馈时退化为 hold，避免乱跑。
                self._control_was_active = False
                return self._hold_action(gripper, trigger, thumbstick, buttons)
            self._robot_origin_pos, self._robot_origin_rot = ee
            self._last_target_pos = self._robot_origin_pos.copy()
            self._last_target_rotvec = self._robot_origin_rot.as_rotvec().copy()
            self._control_was_active = True
            if target_position is None or orientation_quat is None:
                return self._hold_action(gripper, trigger, thumbstick, buttons, enabled=True)

            self._vr_origin_pos = np.asarray(target_position, dtype=float) * self.vr_to_robot_scale
            self._vr_origin_quat = parse_quat_xyzw(orientation_quat)
            return self._emit_target(self._robot_origin_pos, self._robot_origin_rot.as_rotvec(),
                                     gripper, trigger, thumbstick, buttons, enabled=True)

        # 持续激活：计算 target = robot_origin ∘ remap(vr_curr ⊖ vr_origin)。
        if target_position is None or orientation_quat is None or self._vr_origin_pos is None:
            return self._hold_action(gripper, trigger, thumbstick, buttons, enabled=True)

        now = time.perf_counter()
        curr_pos = np.asarray(target_position, dtype=float) * self.vr_to_robot_scale
        curr_quat = parse_quat_xyzw(orientation_quat)
        if self.enable_filter:
            curr_pos = self._pos_filter.filter(curr_pos, now)
            # 四元数逐通道滤波后再归一化(近似，足够降噪用)。
            qf = self._quat_filter.filter(curr_quat, now)
            n = np.linalg.norm(qf)
            curr_quat = qf / n if n > 1e-9 else curr_quat

        # 位置：VR origin body frame 增量 -> remap -> robot origin body frame -> 基坐标系叠加。
        d_pos_vr_body = R.from_quat(self._vr_origin_quat).inv().apply(curr_pos - self._vr_origin_pos)
        d_pos_robot_body = remap_position(d_pos_vr_body, self.axis_remap)
        d_pos_robot_body = self._clip_pos(d_pos_robot_body)
        d_pos_robot_body = self._apply_deadzone_pos(d_pos_robot_body)
        target_pos = self._robot_origin_pos + pos_scale * self._robot_origin_rot.apply(d_pos_robot_body)

        # 姿态：VR origin body frame 旋转增量 -> remap -> 右乘到 robot origin 姿态。
        r_delta_vr = R.from_quat(self._vr_origin_quat).inv() * R.from_quat(curr_quat)
        rotvec_vr = r_delta_vr.as_rotvec()
        rotvec_robot = remap_rotvec(rotvec_vr, self.axis_remap)
        rotvec_robot = self._clip_rotvec(rotvec_robot)
        rotvec_robot = self._apply_deadzone_rotvec(rotvec_robot)
        target_rot = self._robot_origin_rot * R.from_rotvec(angle_scale * rotvec_robot)
        target_rotvec = target_rot.as_rotvec()

        return self._emit_target(target_pos, target_rotvec, gripper, trigger, thumbstick, buttons, enabled=True)

    def _emit_target(
        self,
        target_pos: np.ndarray,
        target_rotvec: np.ndarray,
        gripper: float,
        trigger: float,
        thumbstick: dict,
        buttons: dict,
        *,
        enabled: bool,
    ) -> RobotAction:
        self._last_target_pos = np.asarray(target_pos, dtype=float).copy()
        self._last_target_rotvec = np.asarray(target_rotvec, dtype=float).copy()
        return {
            "ee.enabled": enabled,
            "ee.target_x": float(self._last_target_pos[0]),
            "ee.target_y": float(self._last_target_pos[1]),
            "ee.target_z": float(self._last_target_pos[2]),
            "ee.target_rx": float(self._last_target_rotvec[0]),
            "ee.target_ry": float(self._last_target_rotvec[1]),
            "ee.target_rz": float(self._last_target_rotvec[2]),
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

    def _hold_action(
        self,
        gripper: float,
        trigger: float,
        thumbstick: dict,
        buttons: dict,
        *,
        enabled: bool = False,
    ) -> RobotAction:
        pos = self._last_target_pos if self._last_target_pos is not None else np.zeros(3)
        rv = self._last_target_rotvec if self._last_target_rotvec is not None else np.zeros(3)
        return {
            "ee.enabled": enabled,
            "ee.target_x": float(pos[0]),
            "ee.target_y": float(pos[1]),
            "ee.target_z": float(pos[2]),
            "ee.target_rx": float(rv[0]),
            "ee.target_ry": float(rv[1]),
            "ee.target_rz": float(rv[2]),
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

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in (
            "enabled", "target_position", "orientation_quat",
            "grip_active", "trigger", "thumbstick", "buttons",
        ):
            features[PipelineFeatureType.ACTION].pop(f"xlevr.{feat}", None)

        ee_features = {
            "ee.enabled": FeatureType.STATE,
            "ee.target_x": FeatureType.STATE,
            "ee.target_y": FeatureType.STATE,
            "ee.target_z": FeatureType.STATE,
            "ee.target_rx": FeatureType.STATE,
            "ee.target_ry": FeatureType.STATE,
            "ee.target_rz": FeatureType.STATE,
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
