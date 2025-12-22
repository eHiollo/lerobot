#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import logging
import os
import sys
import time
from typing import Any
from threading import Lock

from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.robots.a10_follower.a10_client import A10TCPClient

from ..teleoperator import Teleoperator
from .config_a10_leader import A10LeaderConfig

logger = logging.getLogger(__name__)

# -------- optional: pynput keyboard listener --------
PYNPUT_AVAILABLE = True
try:
    # On Linux, pynput often requires DISPLAY. If you're on Ubuntu desktop, this should be set.
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        raise ImportError("No DISPLAY set on Linux; pynput may not work in headless mode.")
    from pynput import keyboard
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logger.info(f"[A10Leader] pynput not available: {e}")


class A10Leader(Teleoperator):
    """
    A10 Leader Arm connected via TCP.
    Adds keyboard control for gripper:
      - press 'a' -> open
      - press 'd' -> close
    """

    config_class = A10LeaderConfig
    name = "a10_leader"

    def __init__(self, config: A10LeaderConfig):
        super().__init__(config)
        self.config = config

        # 机械臂关节名（TCP 返回通常是 6 轴；gripper 我们单独键盘控制）
        self.arm_motor_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.gripper_name = "gripper"

        # 对外暴露的 motor_names（包含 gripper，这样 action_features 也包含它）
        self.motor_names = self.arm_motor_names + [self.gripper_name]

        # TCP client（如果你的 controller 只返回 6 个关节角，也没问题）
        self.client = A10TCPClient(
            host=config.host,
            port=config.port,
            joint_names=self.motor_names,  # 保持不改，兼容返回 6 或 7
        )

        # -------- keyboard config (defaults) --------
        # 如果你以后想放到 A10LeaderConfig 里，也可以用 getattr 读取
        self.gripper_open_key = getattr(config, "gripper_open_key", "a")
        self.gripper_close_key = getattr(config, "gripper_close_key", "d")

        # gripper 目标值：你可以按自己夹爪定义改成 0/1 或者角度/行程
        self.gripper_open_pos = float(getattr(config, "gripper_open_pos", 1.0))
        self.gripper_close_pos = float(getattr(config, "gripper_close_pos", 0.0))

        self._gripper_target = self.gripper_open_pos
        self._keys = {self.gripper_open_key: False, self.gripper_close_key: False}
        self._lock = Lock()
        self._kb_listener = None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.client.is_connected

    def _on_key_press(self, key):
        if hasattr(key, "char") and key.char is not None:
            ch = key.char
            if ch in self._keys:
                with self._lock:
                    self._keys[ch] = True

    def _on_key_release(self, key):
        if hasattr(key, "char") and key.char is not None:
            ch = key.char
            if ch in self._keys:
                with self._lock:
                    self._keys[ch] = False

        # ESC 可选退出键盘监听（不影响 TCP 连接）
        if PYNPUT_AVAILABLE and key == keyboard.Key.esc:
            logger.info("[A10Leader] ESC pressed, stopping keyboard listener.")
            self._stop_keyboard_listener()

    def _start_keyboard_listener(self):
        if not PYNPUT_AVAILABLE:
            logger.warning("[A10Leader] pynput not available, keyboard gripper disabled.")
            return
        if self._kb_listener is not None:
            return
        self._kb_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._kb_listener.start()
        logger.info(
            f"[A10Leader] Keyboard enabled: '{self.gripper_open_key}'=OPEN({self.gripper_open_pos}), "
            f"'{self.gripper_close_key}'=CLOSE({self.gripper_close_pos}), ESC=stop keyboard"
        )

    def _stop_keyboard_listener(self):
        if self._kb_listener is not None:
            try:
                self._kb_listener.stop()
            except Exception:
                pass
            self._kb_listener = None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            logger.info(f"{self} already connected.")
        else:
            self.client.connect()
            logger.info(f"{self} connected.")

        # 启动键盘监听（桌面环境可用）
        self._start_keyboard_listener()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._stop_keyboard_listener()
        self.client.disconnect()
        logger.info(f"{self} disconnected.")

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        state_data = self.client.get_state()  # {"q": np.ndarray([...])}
        q = state_data["q"]

        # 兼容两种返回：
        # - len(q) == 6: joint_1..joint_6
        # - len(q) == 7: joint_1..joint_6 + gripper (如果控制器也返回)
        if len(q) not in (6, 7):
            raise ValueError(f"Expected 6 or 7 joints, but got {len(q)} from controller")

        # 先构造 arm action
        action = {f"{name}.pos": float(val) for name, val in zip(self.arm_motor_names, q[:6])}

        # 键盘更新 gripper 目标（按住 a/d 生效，不按保持上一帧）
        with self._lock:
            if self._keys.get(self.gripper_open_key, False):
                self._gripper_target = self.gripper_open_pos
            elif self._keys.get(self.gripper_close_key, False):
                self._gripper_target = self.gripper_close_pos

        # 如果 controller 也返回 gripper，我们允许键盘覆盖（方便你测试）
        action[f"{self.gripper_name}.pos"] = float(self._gripper_target)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        print("get_action 输出", action)
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass
