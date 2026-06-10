#!/usr/bin/env python

"""Quaternion helpers for XLeVR body-frame -> robot EE delta."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / norm


def parse_quat_xyzw(value) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        keys = ("x", "y", "z", "w")
        if not all(k in value for k in keys):
            return None
        return normalize_quat_xyzw([value["x"], value["y"], value["z"], value["w"]])
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape[0] != 4:
        return None
    return normalize_quat_xyzw(arr)


def quat_delta_rotvec_body_rad(q_prev: np.ndarray, q_curr: np.ndarray) -> np.ndarray:
    """Body-frame incremental rotation: R_delta = R_prev^{-1} @ R_curr (scipy xyzw)."""
    r_prev = R.from_quat(normalize_quat_xyzw(q_prev))
    r_curr = R.from_quat(normalize_quat_xyzw(q_curr))
    return (r_prev.inv() * r_curr).as_rotvec()


def delta_position_body(
    p_prev: np.ndarray,
    p_curr: np.ndarray,
    q_prev: np.ndarray,
) -> np.ndarray:
    """Position increment in previous controller body frame (meters, already scaled)."""
    delta = np.asarray(p_curr, dtype=float) - np.asarray(p_prev, dtype=float)
    return R.from_quat(normalize_quat_xyzw(q_prev)).inv().apply(delta)


def remap_position(delta: np.ndarray, axis_remap: tuple[tuple[float, float, float], ...]) -> np.ndarray:
    matrix = np.asarray(axis_remap, dtype=float)
    return matrix @ np.asarray(delta, dtype=float)


def remap_rotvec(rotvec: np.ndarray, axis_remap: tuple[tuple[float, float, float], ...]) -> np.ndarray:
    matrix = np.asarray(axis_remap, dtype=float)
    return matrix @ np.asarray(rotvec, dtype=float)
