"""偏好对数据集：jsonl 格式与合成对构造。

jsonl 每行一条偏好对::

    {"state": [7], "action_a": [[T,7]...], "action_b": [[T,7]...], "label": 1}

``label=1`` 表示 action_a ≻ action_b（a 优于 b）。

真机来源（待 dev/hil 联调）：HIL 介入时刻被否决的 VLA 候选 vs
胜出的人类动作。v1 冷启动可先用合成对（与 RoboMonkey Stage-1 同思路）：
以数据集 ground-truth action 为锚，近邻扰动为正、远扰动为负。
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset


class PreferencePairDataset(Dataset):
    def __init__(self, path: str, action_horizon: int = 10, action_dim: int = 7) -> None:
        self._records: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                a = np.asarray(rec["action_a"], dtype=np.float32)
                b = np.asarray(rec["action_b"], dtype=np.float32)
                if a.shape != (action_horizon, action_dim) or b.shape != (action_horizon, action_dim):
                    raise ValueError(f"第 {lineno} 行 action 形状错误: {a.shape}/{b.shape}")
                self._records.append(rec)
        if not self._records:
            raise ValueError(f"空数据集: {path}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self._records[idx]
        label = int(rec["label"])
        a = torch.as_tensor(np.asarray(rec["action_a"], dtype=np.float32))
        b = torch.as_tensor(np.asarray(rec["action_b"], dtype=np.float32))
        # 对称化：label=0 时交换 a/b 使标签语义统一为 "第一个更优"
        if label == 0:
            a, b = b, a
        return {
            "state": torch.as_tensor(np.asarray(rec["state"], dtype=np.float32)),
            "action_pos": a,
            "action_neg": b,
        }


def make_synthetic_pair(
    anchor: np.ndarray,
    state: np.ndarray,
    noise_near: float = 0.02,
    noise_far: float = 0.2,
    dim_noise_scale: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """以 ground-truth chunk 为锚构造合成偏好对（近扰动正例, 远扰动负例）。

    ``dim_noise_scale``: 可选 (D,) 逐维噪声缩放，用于抹平量纲——A10 前 6 维
    关节为 rad（0.02/0.2 即合理近/远扰动），夹爪为 mm 绝对位置，需放大
    约 100 倍才是等效扰动。
    """
    rng = rng or np.random.default_rng()
    anchor = np.asarray(anchor, dtype=np.float32)
    scale = (
        np.ones(anchor.shape[-1], dtype=np.float32)
        if dim_noise_scale is None
        else np.asarray(dim_noise_scale, dtype=np.float32)
    )
    pos = anchor + rng.normal(size=anchor.shape).astype(np.float32) * (noise_near * scale)
    neg = anchor + rng.normal(size=anchor.shape).astype(np.float32) * (noise_far * scale)
    return {"state": np.asarray(state, dtype=np.float32).tolist(), "action_a": pos.tolist(), "action_b": neg.tolist(), "label": 1}
