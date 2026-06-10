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
            f"  按住右手 squeeze 才控制机械臂，右手摇杆 x 直接控制夹爪",
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

        goal = self._vr_monitor.get_latest_goal_nowait(self.config.arm)
        if goal is None:
            return self._idle_action()

        metadata = goal.metadata or {}
        if metadata.get("reset_target_to_current") and goal.target_position is None:
            return self._idle_action()

        trigger = float(metadata.get("trigger", 0.0))
        buttons = dict(metadata.get("buttons", {}) or {})
        grip_active = bool(metadata.get("grip_active", False))
        if grip_active:
            buttons["squeeze"] = True
        squeeze_active = bool(buttons.get("squeeze", False))

        orientation_quat = parse_quat_xyzw(metadata.get("orientation_quat"))

        return {
            "xlevr.enabled": squeeze_active and goal.target_position is not None,
            "xlevr.target_position": (
                np.asarray(goal.target_position, dtype=float)
                if goal.target_position is not None
                else None
            ),
            "xlevr.orientation_quat": orientation_quat,
            "xlevr.grip_active": grip_active,
            "xlevr.trigger": trigger,
            "xlevr.thumbstick": dict(metadata.get("thumbstick", {}) or {}),
            "xlevr.buttons": buttons,
        }

    def get_vr_events(self) -> dict[str, bool]:
        if self._vr_event_handler is None:
            return {
                "exit_early": False,
                "rerecord_episode": False,
                "stop_recording": False,
            }
        events = self._vr_event_handler.update_events()
        if events.get("exit_early") or events.get("rerecord_episode") or events.get("stop_recording"):
            self._vr_event_handler.reset_events()
        return events

    def _idle_action(self, enabled: bool = False) -> dict[str, Any]:
        return {
            "xlevr.enabled": enabled,
            "xlevr.target_position": None,
            "xlevr.orientation_quat": None,
            "xlevr.trigger": 0.0,
            "xlevr.thumbstick": {},
            "xlevr.buttons": {},
        }

    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass

    def disconnect(self) -> None:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._connected = False
        logger.info("%s disconnected (VR thread runs as daemon).", self)
