"""XLeVRTeleop.get_teleop_events 单元测试 (mock, 不连 VR)。"""
from __future__ import annotations

from lerobot.teleoperators.utils import TeleopEvents
from lerobot.teleoperators.xlevr.teleop_xlevr import XLeVRTeleop


class _StubXLeVR(XLeVRTeleop):
    """绕过 __init__,只测 get_teleop_events 逻辑。"""

    def __init__(self):  # noqa: D401
        self._vr_event_handler = None  # get_vr_events 走 None 分支

    def get_action(self) -> dict:
        return self._fake_action

    def set_fake_action(self, enabled: bool, grip_active: bool) -> None:
        self._fake_action = {"xlevr.enabled": enabled, "xlevr.grip_active": grip_active}


def test_events_no_intervention():
    s = _StubXLeVR()
    s.set_fake_action(enabled=False, grip_active=False)
    ev = s.get_teleop_events()
    assert ev[TeleopEvents.IS_INTERVENTION] is False
    assert ev[TeleopEvents.SUCCESS] is False
    assert ev[TeleopEvents.TERMINATE_EPISODE] is False
    assert ev[TeleopEvents.RERECORD_EPISODE] is False


def test_events_intervention():
    s = _StubXLeVR()
    s.set_fake_action(enabled=True, grip_active=False)
    ev = s.get_teleop_events()
    assert ev[TeleopEvents.IS_INTERVENTION] is True
    assert ev[TeleopEvents.SUCCESS] is False


def test_events_success():
    s = _StubXLeVR()
    s.set_fake_action(enabled=True, grip_active=True)
    ev = s.get_teleop_events()
    assert ev[TeleopEvents.IS_INTERVENTION] is True
    assert ev[TeleopEvents.SUCCESS] is True


if __name__ == "__main__":
    test_events_no_intervention()
    test_events_intervention()
    test_events_success()
    print("XLeVR get_teleop_events 测试通过")
