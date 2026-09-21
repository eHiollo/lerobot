#!/usr/bin/env python

from lerobot.teleoperators.xlevr.hud import (
    build_hud_message,
    display_episode,
    hud_phase_from_record,
    push_record_hud,
    remaining_seconds,
)
from lerobot.teleoperators.xlevr.vr_monitor_bridge import VRMonitorBridge


def test_hud_phase_maps_reset_and_rerecord():
    assert hud_phase_from_record("initial_reset") == "reset"
    assert hud_phase_from_record("recording") == "recording"
    assert hud_phase_from_record("reset", rerecord_episode=True) == "rerecord"
    assert hud_phase_from_record("recording", rerecord_episode=True) == "recording"


def test_display_episode_is_one_based():
    assert display_episode(None) == 1
    assert display_episode(0) == 1
    assert display_episode(4) == 5


def test_remaining_seconds_hides_unlimited_and_invalid_limits():
    assert remaining_seconds(10, 2) == 8.0
    assert remaining_seconds(0, 1) is None
    assert remaining_seconds(float("inf"), 3) is None
    assert remaining_seconds(None, 1) is None


def test_build_hud_message_includes_hint_and_rounded_remaining():
    payload = build_hud_message(
        phase="recording",
        episode=3,
        num_episodes=50,
        remaining_s=61.4,
    )
    assert payload == {
        "type": "hud",
        "phase": "recording",
        "episode": 3,
        "num_episodes": 50,
        "remaining_s": 61,
        "hint": "LEFT = REDO\nUP = RESET\nDOWN = STOP",
    }


def test_push_record_hud_is_a_noop_without_send_hud():
    push_record_hud(object(), phase="reset")


def test_bridge_send_hud_is_a_noop_without_event_loop():
    bridge = VRMonitorBridge("/tmp/xlevr-not-used")
    bridge.send_hud({"type": "hud", "phase": "recording"})
