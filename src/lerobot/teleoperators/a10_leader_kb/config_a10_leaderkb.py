#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

from dataclasses import dataclass
from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("a10_leader_kb")
@dataclass
class A10LeaderKbConfig(TeleoperatorConfig):
    # Host and port to connect to the leader arm via TCP
    host: str
    port: int

    # Backward compatibility with previous policies/dataset
    use_degrees: bool = False

    # ---------- Keyboard gripper control ----------
    # Press 'a' to open, 'd' to close (defaults)
    gripper_open_key: str = "a"
    gripper_close_key: str = "d"

    # Gripper target positions (defaults)
    # Change these to match your actual gripper convention/range
    gripper_open_pos: float = 1.0
    gripper_close_pos: float = 0.0
