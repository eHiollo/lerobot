#!/usr/bin/env python

import json
from types import SimpleNamespace

import pytest

from lerobot.robots.a10_follower.a10_client import A10TCPClient
from lerobot.robots.a10_follower.a10_follower import A10Follower


def anchor_command(**overrides):
    command = {
        "active": True,
        "session_id": "session-a",
        "anchor_id": 3,
        "sample_sequence": 42,
        "offset": [0.1, 0.2, 0.3, 0.01, 0.02, 0.03],
        "gripper": -0.5,
    }
    command.update(overrides)
    return command


def test_client_sends_shell_reset_command():
    client = A10TCPClient("a2.2-reset-test", 18082)
    client.sock = object()
    sent_lines = []
    client._send_line = sent_lines.append

    client.send_reset()

    assert sent_lines == ["reset"]


def test_follower_reset_sends_reset_and_skips_motion():
    follower = object.__new__(A10Follower)
    follower.config = SimpleNamespace(host="127.0.0.1", port=8080, use_ee_delta=True)
    follower.client = FakeClient()
    follower.cameras = {}

    returned = follower.send_action(
        {
            "ee.enabled": True,
            "ee.delta_x": 0.2,
            "gripper.pos": 1.0,
            "_xlevr.reset_arm": True,
        }
    )

    assert follower.client.events == ["reset"]
    assert returned["_xlevr.reset_arm"] is True


def test_client_serializes_set_ee_anchor():
    client = A10TCPClient("a2.2-test", 18080)
    client.sock = object()
    sent_lines = []
    client._send_line = sent_lines.append

    client.send_ee_anchor(anchor_command())

    prefix, raw_payload = sent_lines[0].split(" ", 1)
    assert prefix == "SET_EE_ANCHOR"
    assert json.loads(raw_payload) == anchor_command()


@pytest.mark.parametrize(
    "change",
    (
        {"active": "true"},
        {"session_id": ""},
        {"anchor_id": -1},
        {"sample_sequence": -1},
        {"offset": [0.0] * 5},
        {"offset": [0.0, 0.0, 0.0, 0.0, 0.0, float("nan")]},
    ),
)
def test_client_rejects_invalid_anchor_command(change):
    client = A10TCPClient("a2.2-invalid-test", 18081)
    client.sock = object()
    client._send_line = lambda line: None

    with pytest.raises((TypeError, ValueError)):
        client.send_ee_anchor(anchor_command(**change))


class FakeClient:
    is_connected = True

    def __init__(self):
        self.events = []

    def send_ee_anchor(self, command):
        self.events.append(("anchor", command))

    def send_ee_delta(self, actions):
        self.events.append(("delta", actions))

    def send_reset(self):
        self.events.append("reset")


def test_follower_sends_anchor_first_and_strips_sideband_from_recorded_action():
    follower = object.__new__(A10Follower)
    follower.config = SimpleNamespace(
        use_ee_delta=True,
        send_ee_anchor_shadow=True,
    )
    follower.client = FakeClient()
    follower.cameras = {}
    action = {
        "ee.enabled": True,
        "ee.delta_x": 0.01,
        "ee.delta_y": 0.0,
        "ee.delta_z": 0.0,
        "ee.delta_rx": 0.0,
        "ee.delta_ry": 0.0,
        "ee.delta_rz": 0.0,
        "gripper.pos": 0.25,
        "_xlevr.anchor_command": anchor_command(),
    }

    returned = follower.send_action(action)

    assert [event[0] for event in follower.client.events] == ["anchor", "delta"]
    assert follower.client.events[1][1] == [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]
    assert "_xlevr.anchor_command" not in returned
    assert "_xlevr.anchor_command" in action
