#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.teleoperators.xlevr.vr_events import VREventHandler


@dataclass
class FakeGoal:
    metadata: dict = field(default_factory=dict)


class FakeMonitor:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y

    def get_latest_goal_nowait(self, arm=None):
        return {"left": FakeGoal(metadata={"thumbstick": {"x": self.x, "y": self.y}})}


def test_left_stick_up_requests_arm_reset_without_stopping_recording():
    handler = VREventHandler(FakeMonitor(y=-0.9), threshold=0.7)
    events = handler.update_events()
    assert events["reset_arm"] is True
    assert events["stop_recording"] is False
    assert events["exit_early"] is False
    assert handler.take_reset_arm() is True
    assert handler.take_reset_arm() is False


def test_left_stick_down_still_stops_recording():
    handler = VREventHandler(FakeMonitor(y=0.9), threshold=0.7)
    events = handler.update_events()
    assert events["stop_recording"] is True
    assert events["exit_early"] is True
    assert events["reset_arm"] is False


def test_reset_events_does_not_drop_pending_arm_reset():
    handler = VREventHandler(FakeMonitor(y=-0.9), threshold=0.7)
    handler.update_events()
    handler.reset_events()
    assert handler.take_reset_arm() is True
