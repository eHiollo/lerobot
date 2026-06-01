#!/usr/bin/env python3
"""
Plot recorded VR handle trajectory in 3D from teleoperate.py --record output.

Usage:
    python examples/xlevr_to_a10/plot_vr_trajectory.py recordings/vr_trace_xxx.jsonl
    python examples/xlevr_to_a10/plot_vr_trajectory.py recordings/vr_trace_xxx.jsonl --source cum
    python examples/xlevr_to_a10/plot_vr_trajectory.py recordings/vr_trace_xxx.jsonl --save plot.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_samples(path: Path) -> tuple[list[dict], dict | None]:
    samples: list[dict] = []
    meta: dict | None = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "meta":
                meta = row
            elif row.get("type") == "sample":
                samples.append(row)
    return samples, meta


def pick_xyz(samples: list[dict], source: str) -> tuple[np.ndarray, str]:
    has_vr = all(
        k in s for s in samples for k in ("vr_pos_x", "vr_pos_y", "vr_pos_z")
    )
    if source == "auto":
        source = "vr" if has_vr else "cum"

    if source == "vr":
        keys = ("vr_pos_x", "vr_pos_y", "vr_pos_z")
        label = "VR 手柄绝对位置 (xlevr.target_position)"
    elif source == "cum":
        keys = ("cum_pos_x", "cum_pos_y", "cum_pos_z")
        label = "机器人坐标系积分轨迹 (Σ ee.delta_x/y/z)"
    else:
        raise ValueError(f"Unknown source: {source}")

    xyz = np.array([[float(s[k]) for k in keys] for s in samples], dtype=float)
    return xyz, label


def plot_trajectory(
    samples: list[dict],
    source: str,
    title: str | None,
    save_path: Path | None,
) -> None:
    if not samples:
        raise SystemExit("文件中没有 sample 数据（可能录制时未移动手柄）。")

    xyz, source_label = pick_xyz(samples, source)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#2563eb", linewidth=1.5, alpha=0.85)
    ax.scatter(
        xyz[0, 0],
        xyz[0, 1],
        xyz[0, 2],
        color="#16a34a",
        s=60,
        label="起点",
        depthshade=False,
    )
    ax.scatter(
        xyz[-1, 0],
        xyz[-1, 1],
        xyz[-1, 2],
        color="#dc2626",
        s=60,
        label="终点",
        depthshade=False,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title or f"VR 手柄三维轨迹 ({len(samples)} 点)\n{source_label}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Equal-ish aspect for readability
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1e-4)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"已保存: {save_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot VR trajectory from --record JSONL")
    parser.add_argument("record_file", type=Path, help="teleoperate.py --record 生成的 JSONL 文件")
    parser.add_argument(
        "--source",
        choices=("auto", "vr", "cum"),
        default="auto",
        help="auto=优先 VR 绝对位置，否则用积分轨迹",
    )
    parser.add_argument("--title", default=None, help="图标题")
    parser.add_argument("--save", type=Path, default=None, help="保存 PNG，不弹窗")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.record_file.exists():
        raise SystemExit(f"文件不存在: {args.record_file}")

    samples, meta = load_samples(args.record_file)
    if meta is not None:
        print(
            f"加载 {len(samples)} 个采样点 "
            f"(fps={meta.get('fps', '?')}, started={meta.get('started_at', '?')})"
        )
    plot_trajectory(samples, args.source, args.title, args.save)


if __name__ == "__main__":
    main()
