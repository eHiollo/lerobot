#!/usr/bin/env python

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

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
    trigger: float = 0.0,
):
    return {
        "xlevr.enabled": enabled,
        "xlevr.target_position": np.asarray(position, dtype=float),
        "xlevr.orientation_quat": np.asarray(quat, dtype=float),
        "xlevr.grip_active": enabled,
        "xlevr.trigger": trigger,
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
        "angle_scale": 1.0,
        "pos_deadzone_m": 0.0,
        "angle_deadzone_deg": 0.0,
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


def test_shadow_offset_is_absolute_and_does_not_change_current_action():
    shadow = mapper(compute_anchor_shadow=True)
    baseline = mapper(compute_anchor_shadow=False)
    samples = (
        sample(1, 1.0, [0.0, 0.0, 0.0]),
        sample(2, 1.1, [0.01, 0.0, 0.0]),
        sample(4, 1.2, [0.03, 0.0, 0.0]),
    )

    for action in samples:
        shadow_output = shadow.action(copy.deepcopy(action))
        baseline_output = baseline.action(copy.deepcopy(action))
        anchor_command = shadow_output.pop("_xlevr.anchor_command")
        assert shadow_output == baseline_output

    diagnostics = shadow.get_last_diagnostics()
    np.testing.assert_allclose(
        diagnostics["anchor_shadow_translation_ee_m"], [0.0, 0.03, 0.0]
    )
    np.testing.assert_allclose(translation(shadow_output), [0.0, 0.02, 0.0])
    assert diagnostics["anchor_shadow_transmitted"] is True
    assert diagnostics["anchor_shadow_protocol"] == "SET_EE_ANCHOR/v1"
    assert anchor_command["active"] is True
    assert anchor_command["anchor_id"] == diagnostics["anchor_shadow_id"]
    assert anchor_command["sample_sequence"] == 4
    np.testing.assert_allclose(anchor_command["offset"], diagnostics["anchor_shadow_offset_6d"])


def test_duplicate_sequence_does_not_advance_shadow_offset():
    step = mapper()
    step.action(sample(1, 1.0, [0.0, 0.0, 0.0]))
    step.action(sample(2, 1.1, [0.01, 0.0, 0.0]))
    accepted_offset = step.get_last_diagnostics()["anchor_shadow_offset_6d"]

    output = step.action(sample(2, 1.2, [0.50, 0.0, 0.0]))
    diagnostics = step.get_last_diagnostics()

    assert output["ee.enabled"] is True
    np.testing.assert_allclose(translation(output), np.zeros(3))
    np.testing.assert_allclose(diagnostics["anchor_shadow_offset_6d"], accepted_offset)
    assert diagnostics["anchor_shadow_status"] == "duplicate"


def test_release_and_reengage_create_a_new_zero_offset_anchor():
    step = mapper()
    step.action(sample(1, 1.0, [0.0, 0.0, 0.0]))
    first_anchor = step.get_last_diagnostics()["anchor_shadow_id"]
    step.action(sample(2, 1.1, [0.02, 0.0, 0.0]))

    step.action(sample(3, 1.2, [1.0, 0.0, 0.0], enabled=False))
    released = step.get_last_diagnostics()
    assert released["anchor_shadow_active"] is False

    output = step.action(sample(4, 1.3, [1.0, 0.0, 0.0]))
    reengaged = step.get_last_diagnostics()
    assert output["ee.enabled"] is False
    assert reengaged["anchor_shadow_id"] == first_anchor + 1
    np.testing.assert_allclose(reengaged["anchor_shadow_offset_6d"], np.zeros(6))


def test_anchor_scale_is_frozen_until_reanchor():
    step = mapper(fine_trigger_threshold=0.5, fine_scale_factor=0.5)
    step.action(sample(1, 1.0, [0.0, 0.0, 0.0], trigger=0.0))
    output = step.action(sample(2, 1.1, [0.02, 0.0, 0.0], trigger=1.0))
    diagnostics = step.get_last_diagnostics()

    assert diagnostics["anchor_shadow_pos_scale"] == pytest.approx(1.0)
    np.testing.assert_allclose(
        diagnostics["anchor_shadow_translation_ee_m"], [0.0, 0.02, 0.0]
    )
    np.testing.assert_allclose(translation(output), [0.0, 0.01, 0.0])


def test_anchor_rotation_uses_anchor_body_frame_and_existing_axis_map():
    step = mapper()
    step.action(sample(1, 1.0, [0.0, 0.0, 0.0]))
    quat = R.from_rotvec([0.1, 0.0, 0.0]).as_quat()
    step.action(sample(2, 1.1, [0.0, 0.0, 0.0], quat=quat))
    diagnostics = step.get_last_diagnostics()

    np.testing.assert_allclose(
        diagnostics["anchor_shadow_rotation_ee_rad"], [0.0, 0.1, 0.0], atol=1e-12
    )


def test_confirmed_spike_reanchors_shadow_without_emitting_gap():
    step = mapper(max_vr_speed_m_s=0.5, spike_recovery_frames=2)
    step.action(sample(1, 1.0, [0.0, 0.0, 0.0]))
    first_anchor = step.get_last_diagnostics()["anchor_shadow_id"]

    spike = step.action(sample(2, 1.1, [1.0, 0.0, 0.0]))
    pending = step.get_last_diagnostics()
    reanchor = step.action(sample(3, 1.2, [1.01, 0.0, 0.0]))
    recovered = step.get_last_diagnostics()

    assert spike["ee.enabled"] is False
    assert reanchor["ee.enabled"] is False
    assert pending["anchor_shadow_active"] is False
    assert pending["anchor_shadow_status"] == "spike_pending"
    assert recovered["anchor_shadow_status"] == "reanchored"
    assert recovered["anchor_shadow_id"] == first_anchor + 1
    np.testing.assert_allclose(recovered["anchor_shadow_offset_6d"], np.zeros(6))
