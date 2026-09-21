#!/usr/bin/env python

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_xlevr import XLeVRTeleopConfig
from .quaternion_utils import parse_quat_xyzw
from .vr_events import VREventHandler
from .vr_monitor_bridge import (
    VRMonitorBridge,
    format_port_busy_help,
    get_local_ip,
    is_port_in_use,
)

logger = logging.getLogger(__name__)


class XLeVRTeleop(Teleoperator):
    """
    LeRobot teleoperator backed by XLeVR (WebXR browser VR controllers).

    Outputs raw controller state each frame. A downstream processor converts
    absolute pose streams into `ee.delta_*` commands for the robot controller.
    """

    config_class = XLeVRTeleopConfig
    name = "xlevr"

    def __init__(self, config: XLeVRTeleopConfig):
        super().__init__(config)
        self.config = config
        self._vr_monitor: VRMonitorBridge | None = None
        self._vr_thread: threading.Thread | None = None
        self._connected = False
        self._vr_event_handler: VREventHandler | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "xlevr.enabled": bool,
            "xlevr.target_position": np.ndarray,
            "xlevr.orientation_quat": np.ndarray,
            "xlevr.grip_active": bool,
            "xlevr.trigger": float,
            "xlevr.thumbstick": dict,
            "xlevr.buttons": dict,
            "xlevr.sample_receive_time_s": float,
            "xlevr.sample_age_s": float,
            "xlevr.sample_sequence": int,
            "xlevr.source_timestamp": float,
            "xlevr.reset_arm": bool,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return (
            self._connected
            and self._vr_monitor is not None
            and self._vr_thread is not None
            and self._vr_thread.is_alive()
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")

        xlevr_path = Path(self.config.xlevr_path)
        if not xlevr_path.exists():
            raise FileNotFoundError(f"XLeVR path does not exist: {xlevr_path}")

        self._vr_monitor = VRMonitorBridge(str(xlevr_path))

        init_ok = False
        deadline = time.time() + self.config.connection_timeout
        while time.time() < deadline:
            if self._vr_monitor.initialize():
                init_ok = True
                break
            time.sleep(0.1)

        if not init_ok:
            raise ConnectionError("XLeVR monitor initialization timed out")

        https_port = self._vr_monitor.config.https_port
        ws_port = self._vr_monitor.config.websocket_port
        if is_port_in_use(https_port, self._vr_monitor.config.host_ip):
            raise ConnectionError(format_port_busy_help(https_port, ws_port))

        self._vr_monitor.startup_error = None
        self._vr_thread = threading.Thread(
            target=lambda: asyncio.run(self._vr_monitor.start_monitoring()),
            daemon=True,
        )
        self._vr_thread.start()

        deadline = time.time() + self.config.connection_timeout
        while time.time() < deadline:
            if self._vr_monitor.startup_error is not None:
                err = self._vr_monitor.startup_error
                if isinstance(err, OSError) and err.errno == 98:
                    raise ConnectionError(format_port_busy_help(https_port, ws_port)) from err
                raise ConnectionError(f"XLeVR 服务启动失败: {err}") from err
            if self._vr_monitor.servers_started:
                break
            if not self._vr_thread.is_alive():
                break
            time.sleep(0.1)

        if not self._vr_monitor.servers_started:
            if is_port_in_use(https_port, self._vr_monitor.config.host_ip):
                raise ConnectionError(format_port_busy_help(https_port, ws_port))
            raise ConnectionError("XLeVR monitoring thread failed to start")

        self._connected = True
        if self.config.enable_left_events:
            self._vr_event_handler = VREventHandler(
                self._vr_monitor,
                threshold=self.config.thumbstick_event_threshold,
            )
            self._vr_event_handler.print_control_guide()

        host = get_local_ip()
        https_port = self._vr_monitor.config.https_port
        print(
            f"[XLeVR] Teleoperator 已连接，等待 VR 浏览器...\n"
            f"  打开: https://{host}:{https_port}\n"
            f"  按住右手 squeeze 才控制机械臂，右手摇杆 x 直接控制夹爪\n"
            f"  左手摇杆：右=下一阶段，左=重录，上=机械臂复位，下=停止录制",
            flush=True,
        )
        logger.info("XLeVR ready. Open https://%s:%s in your VR browser.", host, https_port)

    def get_status(self) -> dict[str, Any]:
        if self._vr_monitor is None:
            return {"phase": "not_initialized", "hint": "VR monitor 未初始化"}
        status = self._vr_monitor.get_status()
        status["teleop_connected"] = self._connected
        status["vr_thread_alive"] = self._vr_thread is not None and self._vr_thread.is_alive()
        return status

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        reset_arm = False
        if self._vr_event_handler is not None:
            self._vr_event_handler.update_events()
            reset_arm = self._vr_event_handler.take_reset_arm()

        goal = self._vr_monitor.get_latest_goal_nowait(self.config.arm)
        if goal is None:
            idle = self._idle_action()
            idle["xlevr.reset_arm"] = reset_arm
            return idle

        metadata = goal.metadata or {}
        if metadata.get("reset_target_to_current") and goal.target_position is None:
            idle = self._idle_action()
            idle["xlevr.reset_arm"] = reset_arm
            return idle

        try:
            trigger = float(metadata.get("trigger", 0.0))
        except (TypeError, ValueError):
            trigger = float("nan")
        buttons = dict(metadata.get("buttons", {}) or {})
        grip_active = bool(metadata.get("grip_active", False))
        if grip_active:
            buttons["squeeze"] = True
        squeeze_active = bool(buttons.get("squeeze", False))

        orientation_quat = parse_quat_xyzw(metadata.get("orientation_quat"))
        try:
            receive_time_s = float(metadata.get("pose_receive_monotonic_s", -1.0))
            sample_sequence = int(metadata.get("pose_sequence", -1))
        except (TypeError, ValueError, OverflowError):
            receive_time_s = -1.0
            sample_sequence = -1
        sample_age_s = (
            max(0.0, time.monotonic() - receive_time_s) if receive_time_s >= 0.0 else -1.0
        )
        try:
            target_position = (
                np.asarray(goal.target_position, dtype=float)
                if goal.target_position is not None
                else None
            )
        except (TypeError, ValueError):
            target_position = np.full(3, np.nan)
        source_timestamp = metadata.get("source_timestamp", -1.0)
        try:
            source_timestamp = float(source_timestamp)
        except (TypeError, ValueError):
            source_timestamp = -1.0

        fine_active = trigger >= self.config.fine_trigger_threshold
        if self.config.require_squeeze_to_move:
            control_requested = squeeze_active or fine_active
        else:
            control_requested = True

        return {
            "xlevr.enabled": control_requested and goal.target_position is not None,
            "xlevr.target_position": target_position,
            "xlevr.orientation_quat": orientation_quat,
            "xlevr.grip_active": grip_active,
            "xlevr.trigger": trigger,
            "xlevr.thumbstick": dict(metadata.get("thumbstick", {}) or {}),
            "xlevr.buttons": buttons,
            "xlevr.sample_receive_time_s": receive_time_s,
            "xlevr.sample_age_s": sample_age_s,
            "xlevr.sample_sequence": sample_sequence,
            "xlevr.source_timestamp": source_timestamp,
            "xlevr.reset_arm": reset_arm,
        }

    def get_vr_events(self) -> dict[str, bool]:
        if self._vr_event_handler is None:
            return {
                "exit_early": False,
                "rerecord_episode": False,
                "stop_recording": False,
                "reset_arm": False,
            }
        events = self._vr_event_handler.update_events()
        if events.get("exit_early") or events.get("rerecord_episode") or events.get("stop_recording"):
            self._vr_event_handler.reset_events()
        return events

    def send_hud(self, payload: dict[str, Any]) -> None:
        if not self._connected or self._vr_monitor is None:
            return
        self._vr_monitor.send_hud(payload)

    def _idle_action(self, enabled: bool = False) -> dict[str, Any]:
        return {
            "xlevr.enabled": enabled,
            "xlevr.target_position": None,
            "xlevr.orientation_quat": None,
            "xlevr.grip_active": False,
            "xlevr.trigger": 0.0,
            "xlevr.thumbstick": {},
            "xlevr.buttons": {},
            "xlevr.sample_receive_time_s": -1.0,
            "xlevr.sample_age_s": -1.0,
            "xlevr.sample_sequence": -1,
            "xlevr.source_timestamp": -1.0,
            "xlevr.reset_arm": False,
        }

    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass

    def disconnect(self) -> None:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._connected = False
        logger.info("%s disconnected (VR thread runs as daemon).", self)
