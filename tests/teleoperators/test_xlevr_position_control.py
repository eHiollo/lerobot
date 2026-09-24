#!/usr/bin/env python

import numpy as np
import pytest

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.teleoperators.xlevr.xlevr_processor import XLeVRDeltaEEMapper


IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def sample(
    sequence: int,
    receive_time_s: float,
    position,
    *,
    enabled: bool = True,
    age_s: float = 0.0,
    quat=IDENTITY_QUAT,
):
    return {
        "xlevr.enabled": enabled,
        "xlevr.target_position": np.asarray(position, dtype=float),
        "xlevr.orientation_quat": quat,
        "xlevr.grip_active": enabled,
        "xlevr.trigger": 0.0,
        "xlevr.thumbstick": {},
        "xlevr.buttons": {"squeeze": enabled},
        "xlevr.sample_receive_time_s": receive_time_s,
        "xlevr.sample_age_s": age_s,
        "xlevr.sample_sequence": sequence,
        "xlevr.source_timestamp": -1.0,
    }


def mapper(**kwargs) -> XLeVRDeltaEEMapper:
    defaults = {
        "pos_scale": 1.0,
        "pos_deadzone_m": 0.0,
        "position_cutoff_hz": 0.0,
        "max_vr_speed_m_s": 10.0,
        "max_linear_speed_m_s": 10.0,
        "max_linear_accel_m_s2": 100.0,
        "engage_ramp_s": 0.0,
    }
    defaults.update(kwargs)
    return XLeVRDeltaEEMapper(**defaults)


def translation(output) -> np.ndarray:
    return np.array([output["ee.delta_x"], output["ee.delta_y"], output["ee.delta_z"]])


def test_reset_arm_survives_mapping_and_does_not_move():
    step = mapper()
    action = sample(1, 1.0, [0.0, 0.0, 0.0], enabled=False)
    action["xlevr.reset_arm"] = True

    output = step.action(action)

    assert output["_xlevr.reset_arm"] is True
    assert output["ee.enabled"] is False


def test_first_enable_arms_without_motion_then_preserves_controller_body_mapping():
    step = mapper()

    first = step.action(sample(1, 1.0, [0.0, 0.0, 0.0]))
    second = step.action(sample(2, 1.1, [0.01, 0.0, 0.0]))

    assert first["ee.enabled"] is False
    np.testing.assert_allclose(translation(first), 0.0)
    assert step.control_state == "ACTIVE"
    assert second["ee.enabled"] is True
    np.testing.assert_allclose(translation(second), [0.0, 0.01, 0.0], atol=1e-12)


def test_release_and_reengage_reanchors_without_jump():
    step = mapper()
    step.action(sample(1, 1.0, [0.0, 0.0, 0.0]))
    step.action(sample(2, 1.1, [0.01, 0.0, 0.0]))

    released = step.action(sample(3, 1.2, [0.5, 0.0, 0.0], enabled=False))
    reengaged = step.action(sample(4, 1.3, [0.5, 0.0, 0.0]))

    assert released["ee.enabled"] is False
    assert reengaged["ee.enabled"] is False
    np.testing.assert_allclose(translation(reengaged), 0.0)
    assert step.control_state == "ARMING"


def test_stale_and_zero_quaternion_fail_closed():
    stale_step = mapper(stale_timeout_s=0.2)
    stale = stale_step.action(sample(1, 1.0, [0.0, 0.0, 0.0], age_s=0.21))
    assert stale["ee.enabled"] is False
    assert stale_step.control_state == "STALE"

    invalid_step = mapper()
    invalid = invalid_step.action(sample(1, 1.0, [0.0, 0.0, 0.0], quat=[0, 0, 0, 0]))
    assert invalid["ee.enabled"] is False
    assert invalid_step.control_state == "FAULT"


def test_duplicate_pose_is_not_integrated_twice():
    step = mapper()
    step.action(sample(1, 1.0, [0.0, 0.0, 0.0]))
    moved = step.action(sample(2, 1.1, [0.01, 0.0, 0.0]))
    duplicate = step.action(sample(2, 1.1, [0.01, 0.0, 0.0]))

    assert moved["ee.enabled"] is True
    assert duplicate["ee.enabled"] is True
    np.testing.assert_allclose(translation(duplicate), 0.0)


def test_default_delegates_motion_shaping_to_a10_vr_vel():
    step = mapper(
        max_linear_speed_m_s=0.001,
        max_linear_accel_m_s2=0.001,
        engage_ramp_s=10.0,
    )
    step.action(sample(1, 1.0, [0.0, 0.0, 0.0]))
    output = step.action(sample(2, 1.1, [0.01, 0.0, 0.0]))
    diagnostics = step.get_last_diagnostics()

    np.testing.assert_allclose(translation(output), [0.0, 0.01, 0.0], atol=1e-12)
    assert diagnostics["motion_shaping_mode"] == "robot_controller"
    assert diagnostics["motion_shaping_owner"] == "a10_vr_vel"
    assert diagnostics["shaping_bypassed"] is True
    assert diagnostics["speed_limited"] is False
    assert diagnostics["acceleration_limited"] is False


def test_speed_limit_uses_vector_norm_for_diagonal_motion():
    step = mapper(motion_shaping_mode="lerobot_a1", max_linear_speed_m_s=0.1)
    step.action(sample(1, 1.0, [0.0, 0.0, 0.0]))
    output = step.action(sample(2, 1.1, [0.1, 0.1, 0.0]))

    assert np.linalg.norm(translation(output)) <= 0.1 * 0.1 + 1e-12


def test_acceleration_limit_is_dt_aware():
    step = mapper(
        motion_shaping_mode="lerobot_a1",
        max_linear_accel_m_s2=0.2,
    )
    step.action(sample(1, 1.0, [0.0, 0.0, 0.0]))
    output = step.action(sample(2, 1.1, [0.1, 0.0, 0.0]))

    # max dv = 0.2 * 0.1 = 0.02 m/s; max displacement = dv * dt = 0.002 m
    assert np.linalg.norm(translation(output)) <= 0.002 + 1e-12


def test_spike_requires_confirmed_reanchor_and_never_emits_the_gap():
    step = mapper(max_vr_speed_m_s=0.5, spike_recovery_frames=2)
    step.action(sample(1, 1.0, [0.0, 0.0, 0.0]))

    spike = step.action(sample(2, 1.1, [1.0, 0.0, 0.0]))
    duplicate = step.action(sample(2, 1.1, [1.0, 0.0, 0.0]))
    reanchor = step.action(sample(3, 1.2, [1.01, 0.0, 0.0]))
    recovered = step.action(sample(4, 1.3, [1.02, 0.0, 0.0]))

    for output in (spike, duplicate, reanchor):
        assert output["ee.enabled"] is False
        np.testing.assert_allclose(translation(output), 0.0)
    assert recovered["ee.enabled"] is True
    assert np.linalg.norm(translation(recovered)) < 0.02


def test_legacy_mode_keeps_old_input_contract_without_timestamps():
    step = XLeVRDeltaEEMapper(
        position_control_mode="legacy_frame_delta",
        pos_scale=1.0,
        pos_deadzone_m=0.0,
    )
    first = sample(1, 1.0, [0.0, 0.0, 0.0])
    second = sample(2, 1.1, [0.01, 0.0, 0.0])
    for action in (first, second):
        for key in list(action):
            if key.startswith("xlevr.sample_") or key == "xlevr.source_timestamp":
                action.pop(key)

    step.action(first)
    output = step.action(second)

    assert output["ee.enabled"] is True
    np.testing.assert_allclose(translation(output), [0.0, 0.01, 0.0], atol=1e-12)


def test_sequence_rollback_faults_then_requires_reanchor():
    step = mapper()
    step.action(sample(10, 1.0, [0.0, 0.0, 0.0]))
    step.action(sample(11, 1.1, [0.01, 0.0, 0.0]))

    rollback = step.action(sample(9, 1.2, [0.02, 0.0, 0.0]))
    reanchor = step.action(sample(12, 1.3, [0.02, 0.0, 0.0]))

    assert rollback["ee.enabled"] is False
    assert reanchor["ee.enabled"] is False
    np.testing.assert_allclose(translation(rollback), 0.0)
    np.testing.assert_allclose(translation(reanchor), 0.0)
    assert step.control_state == "ARMING"


def test_diagnostics_explain_arming_active_and_duplicate_frames():
    step = mapper(
        motion_shaping_mode="lerobot_a1",
        max_linear_speed_m_s=0.05,
    )

    first = step.action(sample(1, 1.0, [0.0, 0.0, 0.0]))
    first_diagnostics = step.get_last_diagnostics()
    moved = step.action(sample(2, 1.1, [0.1, 0.0, 0.0]))
    moved_diagnostics = step.get_last_diagnostics()
    step.action(sample(2, 1.1, [0.1, 0.0, 0.0]))
    duplicate_diagnostics = step.get_last_diagnostics()

    assert first_diagnostics["guard_reason"] == "arming"
    assert first_diagnostics["ee_enabled"] is False
    assert moved_diagnostics["guard_reason"] == "active"
    assert moved_diagnostics["sample_dt_s"] == pytest.approx(0.1)
    assert moved_diagnostics["speed_limited"] is True
    assert moved_diagnostics["raw_position_vr_m"] == [0.1, 0.0, 0.0]
    assert moved_diagnostics["final_position_delta_m"] == pytest.approx(
        translation(moved).tolist()
    )
    assert duplicate_diagnostics["guard_reason"] == "duplicate_sample"
    assert not any(key.startswith("vr.") for key in first)
    assert not any(key.startswith("vr.") for key in moved)


def test_diagnostics_explain_stale_invalid_and_spike_frames():
    stale_step = mapper(stale_timeout_s=0.2)
    stale_step.action(sample(1, 1.0, [0.0, 0.0, 0.0], age_s=0.21))
    assert stale_step.get_last_diagnostics()["guard_reason"] == "stale_age"

    invalid_step = mapper()
    invalid_step.action(sample(1, 1.0, [np.nan, 0.0, 0.0]))
    assert invalid_step.get_last_diagnostics()["guard_reason"] == "invalid_sample"

    spike_step = mapper(max_vr_speed_m_s=0.5, spike_recovery_frames=2)
    spike_step.action(sample(1, 1.0, [0.0, 0.0, 0.0]))
    spike_step.action(sample(2, 1.1, [1.0, 0.0, 0.0]))
    assert spike_step.get_last_diagnostics()["guard_reason"] == "spike_pending"


def test_dataset_action_schema_contains_only_robot_command_fields():
    raw_names = (
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
    )
    features = {
        PipelineFeatureType.ACTION: {
            f"xlevr.{name}": PolicyFeature(type=FeatureType.STATE, shape=(1,))
            for name in raw_names
        }
    }

    transformed = mapper().transform_features(features)

    assert set(transformed[PipelineFeatureType.ACTION]) == {
        "ee.enabled",
        "ee.delta_x",
        "ee.delta_y",
        "ee.delta_z",
        "ee.delta_rx",
        "ee.delta_ry",
        "ee.delta_rz",
        "gripper.pos",
    }
