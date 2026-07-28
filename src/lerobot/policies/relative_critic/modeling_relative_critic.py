"""Relative Action Critic 网络与选优接口。

输入 state (7D) 与一对候选 action chunk (T,7)，输出 logit：
正表示 a_i 优于 a_j。选优用锦标赛（tournament）：N 个候选两两比较，
胜场最多者胜出——与 L1 medoid verifier 同为 ``select`` 接口，便于
bridge 端替换。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .configuration_relative_critic import RelativeCriticConfig


class RelativeActionCritic(nn.Module):
    def __init__(self, cfg: RelativeCriticConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_action = cfg.action_horizon * cfg.action_dim

        self.state_enc = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.state_hidden), nn.ReLU(),
            nn.Linear(cfg.state_hidden, cfg.state_hidden), nn.ReLU(),
        )
        self.action_enc = nn.Sequential(
            nn.Linear(in_action, cfg.action_hidden), nn.ReLU(),
            nn.Linear(cfg.action_hidden, cfg.action_hidden), nn.ReLU(),
        )
        pair_in = cfg.state_hidden + 3 * cfg.action_hidden
        self.pair_head = nn.Sequential(
            nn.Linear(pair_in, cfg.pair_hidden), nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.pair_hidden, cfg.pair_hidden), nn.ReLU(),
            nn.Linear(cfg.pair_hidden, 1),
        )

    def forward(self, state: torch.Tensor, action_a: torch.Tensor, action_b: torch.Tensor) -> torch.Tensor:
        """state (B,7), action_a/b (B,T,7) → logit (B,)（正 = a 优于 b）。"""
        s = self.state_enc(state)
        fa = self.action_enc(action_a.flatten(start_dim=1))
        fb = self.action_enc(action_b.flatten(start_dim=1))
        x = torch.cat([s, fa, fb, fa - fb], dim=-1)
        return self.pair_head(x).squeeze(-1)

    @torch.no_grad()
    def win_score(self, state: torch.Tensor, cand_i: torch.Tensor, cand_j: torch.Tensor) -> torch.Tensor:
        return self.forward(state, cand_i, cand_j)

    @torch.no_grad()
    def select(self, state: np.ndarray, candidates: np.ndarray) -> tuple[int, np.ndarray]:
        """锦标赛选优：返回 (best_index, win_counts)。

        state: (7,) 单个观测；candidates: (N,T,7)。两两比较累计胜场。
        """
        self.eval()
        device = next(self.parameters()).device
        c = torch.as_tensor(np.asarray(candidates, dtype=np.float32), device=device)
        s = torch.as_tensor(np.asarray(state, dtype=np.float32), device=device)
        n = c.shape[0]
        if n == 1:
            return 0, np.zeros(1, dtype=np.int64)
        wins = torch.zeros(n, dtype=torch.long, device=device)
        for i in range(n):
            for j in range(i + 1, n):
                logit = self.forward(s.unsqueeze(0), c[i].unsqueeze(0), c[j].unsqueeze(0))
                if bool(logit.item() > 0):
                    wins[i] += 1
                else:
                    wins[j] += 1
        best = int(wins.argmax().item())
        return best, wins.cpu().numpy()
