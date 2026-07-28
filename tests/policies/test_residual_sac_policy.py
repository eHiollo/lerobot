"""ResidualSACPolicy 单元测试 (退化模式, 无 π0.5 base client)。"""
from __future__ import annotations

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.residual_sac import ResidualSACConfig, ResidualSACPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.random_utils import set_seed


def _make_config(state_dim: int = 7, action_dim: int = 7, alpha: float = 0.1) -> ResidualSACConfig:
    cfg = ResidualSACConfig(
        residual_alpha=alpha,
        device="cpu",
        storage_device="cpu",
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        dataset_stats={
            OBS_STATE: {"min": [0.0] * state_dim, "max": [1.0] * state_dim},
            ACTION: {"min": [0.0] * action_dim, "max": [1.0] * action_dim},
        },
    )
    # 关掉 vision encoder (无图像输入)
    cfg.vision_encoder_name = None
    cfg.validate_features()
    return cfg


def test_select_action_degraded():
    """无 base client: a_vla=0, combined = α·Δa ∈ [-α, α]。"""
    set_seed(42)
    cfg = _make_config(alpha=0.1)
    policy = ResidualSACPolicy(cfg)
    policy.eval()
    # 不调用 init_base_client → _chunk_buffer 为 None → 退化模式
    assert policy._chunk_buffer is None

    batch = {OBS_STATE: torch.zeros(1, 7)}
    action = policy.select_action(batch)
    assert action.shape == (1, 7), action.shape
    # 退化模式: combined = 0 + 0.1·Δa, Δa∈[-1,1] → combined∈[-0.1, 0.1]
    assert action.abs().max() <= 0.1 + 1e-5, (action.abs().max(), action)
    print("退化模式 select_action OK, action range:", action.min().item(), action.max().item())


def test_normalize_action():
    """绝对关节 -> [-1,1] 归一化。"""
    set_seed(42)
    cfg = _make_config()
    policy = ResidualSACPolicy(cfg)
    # joint_lower=[-170,-90,-90,-90,-90,-170,0], joint_upper=[170,90,...,100]
    # 中位 [0,0,0,0,0,0,50] -> -1; 上限 -> +1
    mid = torch.tensor([0, 0, 0, 0, 0, 0, 50.0])
    norm_mid = policy._normalize_action(mid)
    assert torch.allclose(norm_mid, torch.zeros(7), atol=1e-4), norm_mid
    upper = torch.tensor([170, 90, 90, 90, 90, 170, 100.0])
    norm_up = policy._normalize_action(upper)
    assert torch.allclose(norm_up, torch.ones(7), atol=1e-4), norm_up
    print("归一化 OK")


def test_reset_clears_chunk():
    set_seed(42)
    cfg = _make_config()
    policy = ResidualSACPolicy(cfg)
    # 无 chunk buffer 时 reset 不报错
    policy.reset()
    print("reset OK")


if __name__ == "__main__":
    test_normalize_action()
    test_reset_clears_chunk()
    test_select_action_degraded()
    print("ResidualSACPolicy 测试全部通过")
