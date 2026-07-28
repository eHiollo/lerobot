"""从 LeRobot 数据集批量生成合成偏好对（L2 critic 冷启动训练数据）。

RoboMonkey Stage-1 式构造：以专家动作 chunk 为锚，近扰动为正例、远扰动为负例。
直接读 parquet（兼容旧版数据集格式，绕过 LeRobotDataset 版本检查），
分维噪声缩放抹平关节 (rad) 与夹爪 (mm) 的量纲差异。

注意：本数据集 ``action`` 列为常量（采集侧未写入），openpi 训练以
``use_state_as_action_targets=True`` 从 ``observation.state`` 未来序列构造动作；
本脚本遵循同一约定，锚 chunk 取 ``state[t:t+T]`` 绝对关节序列——与 bridge 端
candidates 的空间一致（推理输出经 AbsoluteActions 变换后同样是绝对关节）。

用法::

    python -m lerobot.policies.relative_critic.generate_synthetic_pairs \\
        --data-dir /home/allen/Allen/openpi/dataset/dataset_5_9 \\
        --output pairs_5_9.jsonl --stride 2 --pairs-per-step 3 --max-pairs 20000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .dataset_preference import make_synthetic_pair


def _load_states(path: Path) -> np.ndarray:
    """返回 states [F, D]（绝对关节）。"""
    t = pq.read_table(path, columns=["observation.state"])
    return np.stack(t.column("observation.state").to_pylist()).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic preference pairs from a LeRobot dataset (parquet).")
    parser.add_argument("--data-dir", type=str, required=True, help="LeRobot 数据集根目录 (含 data/chunk-*/)")
    parser.add_argument("--output", type=str, required=True, help="输出 jsonl 路径")
    parser.add_argument("--horizon", type=int, default=10, help="锚 chunk 长度 T")
    parser.add_argument("--stride", type=int, default=2, help="每 episode 采样步距（相邻 chunk 重叠大, 不必逐帧采）")
    parser.add_argument("--pairs-per-step", type=int, default=1, help="每时间步生成的对数（不同噪声种子, 数据增广）")
    parser.add_argument("--noise-near", type=float, default=0.02, help="正例噪声幅度 (rad 尺度)")
    parser.add_argument("--noise-far", type=float, default=0.2, help="负例噪声幅度 (rad 尺度)")
    parser.add_argument("--gripper-noise-scale", type=float, default=100.0, help="夹爪维噪声放大倍数 (rad→mm)")
    parser.add_argument("--gripper-dim", type=int, default=6, help="夹爪维度索引")
    parser.add_argument("--max-pairs", type=int, default=20000, help="最多输出对数（全局打乱后截断）")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    files = sorted(Path(args.data_dir).glob("data/chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode parquet under {args.data_dir}/data/chunk-*/")

    rng = np.random.default_rng(args.seed)
    pairs: list[dict] = []
    for f in files:
        states = _load_states(f)
        n_frames, dim = states.shape
        scale = np.ones(dim, dtype=np.float32)
        scale[args.gripper_dim] = args.gripper_noise_scale
        for t0 in range(0, n_frames - args.horizon, args.stride):
            anchor = states[t0 : t0 + args.horizon]  # 未来绝对 state 序列 = 模型动作目标空间
            for _ in range(args.pairs_per_step):
                pairs.append(
                    make_synthetic_pair(
                        anchor,
                        states[t0],
                        noise_near=args.noise_near,
                        noise_far=args.noise_far,
                        dim_noise_scale=scale,
                        rng=rng,
                    )
                )

    rng.shuffle(pairs)
    pairs = pairs[: args.max_pairs]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[generate_synthetic_pairs] {len(files)} episodes -> {len(pairs)} pairs -> {out}")


if __name__ == "__main__":
    main()
