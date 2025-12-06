from dataclasses import dataclass

from lerobot.teleoperators.teleoperator import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("x7_leader")
@dataclass
class X7LeaderConfig(TeleoperatorConfig):
    host: str = "127.0.0.1"
    port: int = 8000
    n_joints: int = 8
