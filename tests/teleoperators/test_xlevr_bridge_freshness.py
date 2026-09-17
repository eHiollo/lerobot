#!/usr/bin/env python

from dataclasses import dataclass, field

import numpy as np

from lerobot.teleoperators.xlevr.vr_monitor_bridge import VRMonitorBridge


@dataclass
class FakeGoal:
    arm: str = "right"
    target_position: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


def test_partial_goal_does_not_make_an_old_pose_look_fresh():
    bridge = VRMonitorBridge("/tmp/xlevr-not-used")
    pose = FakeGoal(
        target_position=np.array([1.0, 2.0, 3.0]),
        metadata={"orientation_quat": [0.0, 0.0, 0.0, 1.0], "timestamp": 1234.0},
    )
    bridge._record_goal(pose, receive_monotonic_s=10.0, receive_wall_s=100.0)

    partial = FakeGoal(metadata={"trigger": 0.8})
    merged = bridge._record_goal(partial, receive_monotonic_s=11.0, receive_wall_s=101.0)

    np.testing.assert_allclose(merged.target_position, [1.0, 2.0, 3.0])
    assert merged.metadata["bridge_goal_sequence"] == 2
    assert merged.metadata["bridge_receive_monotonic_s"] == 11.0
    assert merged.metadata["pose_sequence"] == 1
    assert merged.metadata["pose_receive_monotonic_s"] == 10.0
    assert merged.metadata["source_timestamp"] == 1234.0
    assert bridge.last_pose_monotonic_s == 10.0


def test_new_pose_advances_pose_sequence_and_receive_time():
    bridge = VRMonitorBridge("/tmp/xlevr-not-used")
    first = FakeGoal(target_position=np.zeros(3), metadata={"timestamp": 1234.0})
    second = FakeGoal(target_position=np.ones(3))

    bridge._record_goal(first, receive_monotonic_s=10.0, receive_wall_s=100.0)
    recorded = bridge._record_goal(second, receive_monotonic_s=10.1, receive_wall_s=100.1)

    assert recorded.metadata["pose_sequence"] == 2
    assert recorded.metadata["pose_receive_monotonic_s"] == 10.1
    assert bridge.goals_received == 2
    assert recorded.metadata["source_timestamp"] is None
    assert bridge.last_goal_monotonic_s == 10.1
    assert bridge.last_pose_monotonic_s == 10.1


def test_status_reports_monotonic_goal_and_pose_age(monkeypatch):
    bridge = VRMonitorBridge("/tmp/xlevr-not-used")
    bridge._record_goal(
        FakeGoal(target_position=np.zeros(3)),
        receive_monotonic_s=10.0,
        receive_wall_s=100.0,
    )
    monkeypatch.setattr(
        "lerobot.teleoperators.xlevr.vr_monitor_bridge.time.monotonic", lambda: 10.2
    )

    status = bridge.get_status()

    assert abs(status["last_goal_age_s"] - 0.2) < 1e-12
    assert abs(status["last_pose_age_s"] - 0.2) < 1e-12
