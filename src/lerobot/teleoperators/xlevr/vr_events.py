#!/usr/bin/env python

import logging

logger = logging.getLogger(__name__)


class VREventHandler:
    """
    Map left-controller thumbstick to LeRobot recording events.

    Left thumbstick:
      - right  -> exit current episode early
      - left   -> re-record episode
      - up     -> stop recording
    """

    def __init__(self, vr_monitor, threshold: float = 0.7):
        self.vr_monitor = vr_monitor
        self.threshold = threshold
        self.events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
        }
        self._prev = {"x": 0.0, "y": 0.0}

    def update_events(self) -> dict[str, bool]:
        dual = self.vr_monitor.get_latest_goal_nowait()
        left_goal = dual.get("left") if isinstance(dual, dict) else None
        if left_goal is None or not getattr(left_goal, "metadata", None):
            return self.events.copy()

        thumb = left_goal.metadata.get("thumbstick", {}) or {}
        x = float(thumb.get("x", 0.0))
        y = float(thumb.get("y", 0.0))

        if x > self.threshold and self._prev["x"] <= self.threshold:
            logger.info("VR left thumbstick right -> exit_early")
            self.events["exit_early"] = True
        elif x < -self.threshold and self._prev["x"] >= -self.threshold:
            logger.info("VR left thumbstick left -> rerecord_episode")
            self.events["rerecord_episode"] = True
            self.events["exit_early"] = True
        if y > self.threshold and self._prev["y"] <= self.threshold:
            logger.info("VR left thumbstick up -> stop_recording")
            self.events["stop_recording"] = True
            self.events["exit_early"] = True

        self._prev = {"x": x, "y": y}
        return self.events.copy()

    def reset_events(self) -> None:
        for key in self.events:
            self.events[key] = False

    def print_control_guide(self) -> None:
        logger.info(
            "VR recording controls (left thumbstick): "
            "right=next episode, left=re-record, up=stop"
        )
