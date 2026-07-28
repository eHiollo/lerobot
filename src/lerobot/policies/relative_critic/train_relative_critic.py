"""Relative Action Critic 训练脚本。

用法::

    python -m lerobot.policies.relative_critic.train_relative_critic \
      --data preference_pairs.jsonl --output outputs/relative_critic/run1

数据格式见 dataset_preference.py。合成冷启动数据可由
``make_synthetic_pair`` 从 LeRobot 数据集批量生成（后续接 HIL 真实对）。
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .configuration_relative_critic import RelativeCriticConfig
from .dataset_preference import PreferencePairDataset
from .modeling_relative_critic import RelativeActionCritic

logger = logging.getLogger(__name__)


def evaluate(model: RelativeActionCritic, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            s = batch["state"].to(device)
            pos = batch["action_pos"].to(device)
            neg = batch["action_neg"].to(device)
            logit = model(s, pos, neg)
            correct += int((logit > 0).sum().item())
            total += logit.shape[0]
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 Relative Action Critic")
    parser.add_argument("--data", type=str, required=True, help="偏好对 jsonl")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = RelativeCriticConfig()
    device = cfg.device if torch.cuda.is_available() else "cpu"

    dataset = PreferencePairDataset(args.data, action_horizon=cfg.action_horizon, action_dim=cfg.action_dim)
    n_val = max(1, int(len(dataset) * cfg.val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size)

    model = RelativeActionCritic(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("train=%d val=%d device=%s", n_train, n_val, device)
    step = 0
    best_acc = 0.0
    t_start = time.time()
    while step < cfg.max_steps:
        for batch in train_loader:
            model.train()
            s = batch["state"].to(device)
            pos = batch["action_pos"].to(device)
            neg = batch["action_neg"].to(device)
            logit = model(s, pos, neg)
            loss = loss_fn(logit, torch.ones_like(logit))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % cfg.log_every == 0:
                acc = evaluate(model, val_loader, device)
                logger.info("step=%d loss=%.4f val_acc=%.3f elapsed=%.0fs", step, loss.item(), acc, time.time() - t_start)
                if acc > best_acc:
                    best_acc = acc
                    torch.save({"model": model.state_dict(), "config": vars(cfg), "val_acc": acc}, out_dir / "best.pt")
            if step >= cfg.max_steps:
                break

    torch.save({"model": model.state_dict(), "config": vars(cfg)}, out_dir / "last.pt")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_acc": best_acc, "steps": step}, f, indent=2)
    logger.info("完成: best_val_acc=%.3f, 输出 %s", best_acc, out_dir)


if __name__ == "__main__":
    main()
