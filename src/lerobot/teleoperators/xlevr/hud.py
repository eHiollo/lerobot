"""Headset HUD payloads for XLeVR recording phases."""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

HUD_HINTS = {
    "reset": "RIGHT = START\nUP = RESET\nDOWN = STOP",
    "initial_reset": "RIGHT = START\nUP = RESET\nDOWN = STOP",
    "recording": "LEFT = REDO\nUP = RESET\nDOWN = STOP",
    "rerecord": "RIGHT = START\nUP = RESET\nDOWN = STOP",
    "stop": "",
}


def hud_phase_from_record(diagnostics_phase: str, rerecord_episode: bool = False) -> str:
    if rerecord_episode and diagnostics_phase in ("reset", "initial_reset"):
        return "rerecord"
    if diagnostics_phase == "initial_reset":
        return "reset"
    return diagnostics_phase or "reset"


def display_episode(diagnostics_episode_index: int | None) -> int:
    if diagnostics_episode_index is None:
        return 1
    return int(diagnostics_episode_index) + 1


def remaining_seconds(control_time_s: float | None, elapsed_s: float) -> float | None:
    if control_time_s is None:
        return None
    try:
        limit = float(control_time_s)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(limit) or limit <= 0:
        return None
    return max(0.0, limit - max(0.0, float(elapsed_s)))


def build_hud_message(
    *,
    phase: str,
    episode: int | None = None,
    num_episodes: int | None = None,
    remaining_s: float | None = None,
    hint: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "hud", "phase": str(phase)}
    if episode is not None:
        payload["episode"] = int(episode)
    if num_episodes is not None:
        payload["num_episodes"] = int(num_episodes)
    if remaining_s is not None:
        try:
            remaining = float(remaining_s)
        except (TypeError, ValueError):
            remaining = float("nan")
        if math.isfinite(remaining):
            payload["remaining_s"] = int(round(max(0.0, remaining)))
    resolved_hint = HUD_HINTS.get(phase, "") if hint is None else hint
    if resolved_hint:
        payload["hint"] = resolved_hint
    return payload


def push_record_hud(teleop: Any, **kwargs: Any) -> None:
    sender = getattr(teleop, "send_hud", None)
    if sender is None:
        return
    try:
        sender(build_hud_message(**kwargs))
    except Exception:
        logger.debug("VR HUD push failed", exc_info=True)
