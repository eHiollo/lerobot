"""ResidualSAC 策略:冻结 π0.5 + 小 SAC 残差头。

action = normalize(a_vla) + α · Δa
  - a_vla: π0.5 websocket 服务返回的绝对关节 7D (冻结, 不反传)
  - Δa:    SAC actor 输出 [-1,1] 残差
  - α:     residual_alpha

推理 (select_action):
  1. 从 π0.5 服务拉 a_vla (chunk 逐步释放)
  2. 用 joint_lower/upper 把 a_vla 归一化到 [-1,1]
  3. SAC actor 在处理后的 batch 上输出 Δa
  4. 返回 clip(a_vla_norm + α·Δa, -1, 1)

训练 (forward): 复用 SAC 的 critic loss (buffer 中存的是 combined action);
actor loss 改为最大化 Q(s, a_vla_norm + α·Δa),a_vla detach。
完整训练集成见 Phase 4 learner.py 接线。
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import torch
from torch import nn

from lerobot.configs.types import FeatureType
from lerobot.utils.constants import ACTION

from ..sac.configuration_sac import SACConfig
from ..sac.modeling_sac import SACPolicy
from .configuration_residual_sac import ResidualSACConfig

logger = logging.getLogger(__name__)

# openpi-client 路径 (不污染 lerobot 依赖)
_OPENPI_CLIENT = "/home/allen/Allen/openpi/packages/openpi-client/src"


def _ensure_openpi_client() -> None:
    if _OPENPI_CLIENT not in sys.path:
        sys.path.insert(0, _OPENPI_CLIENT)


class _ChunkBuffer:
    """逐步释放 π0.5 action chunk;耗尽时重新推理。"""

    def __init__(self, client, horizon: int) -> None:
        self._client = client
        self._horizon = horizon
        self._chunk: np.ndarray | None = None
        self._step = 0

    def reset(self) -> None:
        self._chunk = None
        self._step = 0

    def get(self, obs: dict) -> np.ndarray:
        """返回当前步的 7D 绝对关节;必要时重新推理。"""
        if self._chunk is None:
            result = self._client.infer(obs)
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None, :]
            elif actions.ndim == 3 and actions.shape[0] == 1:
                actions = actions[0]
            self._chunk = np.ascontiguousarray(actions, dtype=np.float32)
            self._step = 0
        action = self._chunk[self._step]
        self._step += 1
        if self._step >= self._horizon or self._step >= len(self._chunk):
            self._chunk = None
        return action


class ResidualSACPolicy(SACPolicy):
    """残差 SAC:在冻结 π0.5 之上叠加小残差头。

    继承 SACPolicy 以复用其 encoder/critic/actor/temperature 与 loss 机器,
    仅覆盖 select_action (叠加 a_vla) 与 reset (清 chunk buffer)。
    """

    config_class = ResidualSACConfig
    name = "residual_sac"

    def __init__(self, config: ResidualSACConfig | None = None):
        # 用 SAC 子配置初始化父类 (复用全部 SAC 网络)
        super().__init__(config)

        # π0.5 base policy 客户端 (懒加载, 真机/训练时才连)
        self._base_client = None
        self._chunk_buffer: _ChunkBuffer | None = None
        self._base_prompt = config.base_policy_prompt if config else "Reach the yellow lemon"
        self._alpha = config.residual_alpha if config else 0.1
        jl = config.joint_lower if config else [-170.0] * 6 + [0.0]
        ju = config.joint_upper if config else [170.0] * 6 + [100.0]
        self._joint_lower = torch.tensor(jl, dtype=torch.float32)
        self._joint_upper = torch.tensor(ju, dtype=torch.float32)

    # --- π0.5 客户端管理 ---

    def init_base_client(self) -> None:
        """显式初始化 π0.5 websocket 客户端 (actor 启动时调用)。"""
        _ensure_openpi_client()
        from openpi_client import websocket_client_policy

        self._base_client = websocket_client_policy.WebsocketClientPolicy(
            host=self.config.base_policy_host,
            port=self.config.base_policy_port,
        )
        self._chunk_buffer = _ChunkBuffer(self._base_client, self.config.base_action_horizon)
        logger.info(
            "ResidualSAC base client 已连接 %s:%d (α=%g)",
            self.config.base_policy_host, self.config.base_policy_port, self._alpha,
        )

    def reset(self) -> None:
        """新 episode 开始时清 chunk buffer。"""
        if self._chunk_buffer is not None:
            self._chunk_buffer.reset()

    # --- 归一化 ---

    def _normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        """绝对关节 -> [-1,1]。a: (..., 7)"""
        lower = self._joint_lower.to(a.device)
        upper = self._joint_upper.to(a.device)
        return 2.0 * (a - lower) / (upper - lower) - 1.0

    # --- 推理 ---

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], base_obs: dict | None = None) -> torch.Tensor:
        """叠加 π0.5 base action 与 SAC 残差。

        Args:
            batch: SAC 处理后的观测 batch (含 observation.state, observation.images.*)
            base_obs: π0.5 需要的原始观测 (state, images/right, images/top, prompt);
                      为 None 时 a_vla=0 (退化模式, 仅用于离线单测)
        Returns:
            combined action (batch, 7) in [-1,1]
        """
        # 1. SAC 残差 Δa
        observations_features = None
        if self.shared_encoder and self.actor.encoder.has_images:
            observations_features = self.actor.encoder.get_cached_image_features(batch)
        delta_a, _, _ = self.actor(batch, observations_features)  # (B, 7) [-1,1]

        # 2. π0.5 base action a_vla
        if self._chunk_buffer is not None and base_obs is not None:
            a_vla_abs = self._chunk_buffer.get(base_obs)  # (7,) 绝对关节
            a_vla_norm = self._normalize_action(
                torch.as_tensor(a_vla_abs, dtype=torch.float32, device=delta_a.device)
            )
            a_vla_norm = a_vla_norm.view(1, -1)  # (1, 7)
            # 广播到 batch
            if delta_a.shape[0] > 1:
                a_vla_norm = a_vla_norm.expand(delta_a.shape[0], -1)
        else:
            # 退化模式:无 base policy,纯 SAC (用于离线单测/调试)
            a_vla_norm = torch.zeros_like(delta_a)

        # 3. combined = a_vla + α·Δa
        combined = torch.clamp(a_vla_norm + self._alpha * delta_a, -1.0, 1.0)
        return combined

    # --- 训练 (Phase 4 TODO) ---
    # forward() 复用父类 SACPolicy.forward (critic loss 不变, buffer 存 combined action)。
    # actor loss 需改为: maximize Q(s, a_vla_norm + α·Δa), a_vla detach。
    # 该修改在 Phase 4 接入 learner.py 时实现 (需在 batch 中携带 a_vla_norm)。
