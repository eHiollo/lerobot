"""ResidualSAC 配置:冻结 π0.5 + 小 SAC 残差头。

继承 SACConfig 以复用全部 SAC 超参 (encoder/critic/actor/temperature/utd...),
新增 π0.5 base policy 连接 + 残差缩放字段。

残差 RL: action = normalize(a_vla) + α · Δa
  - a_vla: π0.5 (冻结, 外部 websocket 服务) 输出的绝对关节, 归一化到 [-1,1]
  - Δa:    SAC actor 输出的残差 [-1,1]
  - α:     residual_alpha
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import OBS_IMAGE, OBS_STATE

from ..sac.configuration_sac import SACConfig


@PreTrainedConfig.register_subclass("residual_sac")
@dataclass
class ResidualSACConfig(SACConfig):
    """残差 SAC 配置 (继承 SAC,叠加在冻结 π0.5 之上)。"""

    # --- π0.5 base policy 服务 ---
    base_policy_host: str = "127.0.0.1"
    base_policy_port: int = 8000
    base_policy_prompt: str = "Reach the yellow lemon"
    base_action_horizon: int = 10  # π0.5 chunk 长度
    base_action_dim: int = 7

    # --- 残差 ---
    residual_alpha: float = 0.1  # action = a_vla_norm + α·Δa
    # 关节限位 (用于把 π0.5 绝对关节归一化到 [-1,1]);与 A10RobotEnv 一致
    joint_lower: list[float] = field(
        default_factory=lambda: [-170.0, -90.0, -90.0, -90.0, -90.0, -170.0, 0.0]
    )
    joint_upper: list[float] = field(
        default_factory=lambda: [170.0, 90.0, 90.0, 90.0, 90.0, 170.0, 100.0]
    )
