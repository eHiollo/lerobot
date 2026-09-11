#!/usr/bin/env python

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.teleoperators.xlevr.quaternion_utils import (
    AXIS_REMAP_VR_TO_ROBOT,
    remap_position,
    remap_rotvec,
)


def test_axis_remap_is_so3():
    m = np.asarray(AXIS_REMAP_VR_TO_ROBOT, dtype=float)
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(m), 1.0, atol=1e-12)


def test_translation_ee_directions():
    """VR body +X right, +Y up, +Z back → EE +X up, +Y right, +Z forward."""
    m = AXIS_REMAP_VR_TO_ROBOT
    np.testing.assert_allclose(remap_position([0.0, 1.0, 0.0], m), [1.0, 0.0, 0.0])  # up → +X
    np.testing.assert_allclose(remap_position([1.0, 0.0, 0.0], m), [0.0, 1.0, 0.0])  # right → +Y
    np.testing.assert_allclose(remap_position([0.0, 0.0, -1.0], m), [0.0, 0.0, 1.0])  # forward → +Z


def test_rpy_follows_ee_axes():
    m = AXIS_REMAP_VR_TO_ROBOT
    np.testing.assert_allclose(remap_rotvec([0.0, 1.0, 0.0], m), [1.0, 0.0, 0.0])  # VR yaw-up → rx
    np.testing.assert_allclose(remap_rotvec([1.0, 0.0, 0.0], m), [0.0, 1.0, 0.0])  # VR pitch-right → ry
    np.testing.assert_allclose(remap_rotvec([0.0, 0.0, -1.0], m), [0.0, 0.0, 1.0])  # VR roll-forward → rz


def test_finite_rotvec_is_change_of_basis():
    m = np.asarray(AXIS_REMAP_VR_TO_ROBOT, dtype=float)
    rotvec_vr = np.array([0.3, -0.4, 0.5])
    rotvec_robot = remap_rotvec(rotvec_vr, AXIS_REMAP_VR_TO_ROBOT)
    r_vr = R.from_rotvec(rotvec_vr)
    r_expected = R.from_matrix(m @ r_vr.as_matrix() @ m.T)
    np.testing.assert_allclose(rotvec_robot, r_expected.as_rotvec(), atol=1e-12)
