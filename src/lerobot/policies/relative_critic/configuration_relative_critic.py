"""Relative Action Critic 配置。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RelativeCriticConfig:
    # --- 数据维度 ---
    state_dim: int = 7
    action_dim: int = 7
    action_horizon: int = 10

    # --- 网络 ---
    state_hidden: int = 128
    action_hidden: int = 128
    pair_hidden: int = 256
    dropout: float = 0.1

    # --- 训练 ---
    lr: float = 3e-4
    batch_size: int = 64
    max_steps: int = 20_000
    weight_decay: float = 1e-4
    val_split: float = 0.1
    log_every: int = 100
    device: str = "cuda"
