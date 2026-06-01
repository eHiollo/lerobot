#!/usr/bin/env python3
"""Record VR teleop samples when end-effector deltas are non-zero."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

EE_DELTA_KEYS = (
    "ee.delta_x",
    "ee.delta_y",
    "ee.delta_z",
    "ee.delta_roll",
    "ee.delta_pitch",
    "ee.delta_yaw",
)


def has_ee_motion(action: dict[str, Any], eps: float = 1e-12) -> bool:
    """True if any of the six ee.delta_* values is non-zero."""
    return any(abs(float(action.get(key, 0.0))) > eps for key in EE_DELTA_KEYS)


def _vr_position(raw_action: dict[str, Any]) -> list[float] | None:
    pos = raw_action.get("xlevr.target_position")
    if pos is None:
        return None
    arr = np.asarray(pos, dtype=float).reshape(-1)
    if arr.size < 3:
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


class VRTrajectoryRecorder:
    """Append JSONL rows while the operator moves the controller."""

    def __init__(self, output_path: Path, fps: int) -> None:
        self.output_path = output_path
        self.fps = fps
        self._file = output_path.open("w", encoding="utf-8")
        self._cum_pos = np.zeros(3, dtype=float)
        self._sample_count = 0
        self._started_at = time.time()

        self._write_meta()

    def _write_meta(self) -> None:
        meta = {
            "type": "meta",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "fps": self.fps,
            "delta_keys": list(EE_DELTA_KEYS),
            "note": "cum_pos_* integrates ee.delta_x/y/z in robot frame; vr_pos_* is raw VR handle position",
        }
        self._file.write(json.dumps(meta, ensure_ascii=False) + "\n")
        self._file.flush()

    def maybe_record(
        self,
        frame: int,
        robot_action: dict[str, Any],
        raw_action: dict[str, Any],
    ) -> bool:
        if not has_ee_motion(robot_action):
            return False

        self._cum_pos[0] += float(robot_action.get("ee.delta_x", 0.0))
        self._cum_pos[1] += float(robot_action.get("ee.delta_y", 0.0))
        self._cum_pos[2] += float(robot_action.get("ee.delta_z", 0.0))

        row: dict[str, Any] = {
            "type": "sample",
            "frame": frame,
            "timestamp": time.time() - self._started_at,
            **{key: float(robot_action.get(key, 0.0)) for key in EE_DELTA_KEYS},
            "cum_pos_x": float(self._cum_pos[0]),
            "cum_pos_y": float(self._cum_pos[1]),
            "cum_pos_z": float(self._cum_pos[2]),
            "gripper.pos": float(robot_action.get("gripper.pos", 0.0)),
            "vr.button_squeeze": bool(robot_action.get("vr.button_squeeze", False)),
        }

        vr_pos = _vr_position(raw_action)
        if vr_pos is not None:
            row["vr_pos_x"] = vr_pos[0]
            row["vr_pos_y"] = vr_pos[1]
            row["vr_pos_z"] = vr_pos[2]

        self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._file.flush()
        self._sample_count += 1
        return True

    def close(self) -> int:
        if self._file.closed:
            return self._sample_count
        self._file.write(
            json.dumps(
                {
                    "type": "footer",
                    "sample_count": self._sample_count,
                    "duration_s": time.time() - self._started_at,
                    "output": str(self.output_path),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self._file.flush()
        self._file.close()
        return self._sample_count
