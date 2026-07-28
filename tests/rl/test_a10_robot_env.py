"""A10RobotEnv 单元测试 (mock 机器人, 不连真机)。"""
from __future__ import annotations

import numpy as np

from lerobot.rl.gym_manipulator_a10 import A10RobotEnv, _denormalize, DEFAULT_JOINT_LOWER, DEFAULT_JOINT_UPPER


class MockA10Robot:
    """模拟 A10Follower:关节 + 单相机,记录 send_action。"""

    def __init__(self, n_joints: int = 7):
        self.joint_names = [f"joint_{i}" for i in range(1, n_joints)] + ["gripper"]
        self.cameras = {"right_wrist_0_rgb": _MockCam()}
        self._q = np.zeros(n_joints, dtype=np.float32)
        self._q[-1] = 50.0  # gripper 中位
        self._connected = False
        self.sent_actions: list[dict] = []

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_observation(self) -> dict:
        out = {f"{n}.pos": float(self._q[i]) for i, n in enumerate(self.joint_names)}
        out["right_wrist_0_rgb"] = self.cameras["right_wrist_0_rgb"].read()
        return out

    def send_action(self, action: dict) -> dict:
        self.sent_actions.append(action)
        for i, n in enumerate(self.joint_names):
            k = f"{n}.pos"
            if k in action:
                self._q[i] = action[k]
        return action


class _MockCam:
    def __init__(self) -> None:
        self.is_connected = True

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def read(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)


def test_env_basic():
    robot = MockA10Robot()
    reset_pose = [0, 0, 0, 0, 0, 0, 50.0]
    env = A10RobotEnv(robot=robot, reset_pose=reset_pose, reset_time_s=0.01)

    # 观测空间
    assert OBS_STATE in env.observation_space.spaces
    assert f"{OBS_IMAGES}.right_wrist_0_rgb" in env.observation_space.spaces
    assert env.action_space.shape == (7,)
    assert env.action_space.low.shape == (7,)

    # reset
    obs, info = env.reset()
    assert obs["agent_pos"].shape == (7,)
    assert "pixels" in obs and "right_wrist_0_rgb" in obs["pixels"]
    assert info[TeleopEvents.IS_INTERVENTION] is False
    # reset 应该发了多步插值动作
    assert len(robot.sent_actions) > 0

    # step: 中位动作
    sent_before = len(robot.sent_actions)
    action = np.zeros(7, dtype=np.float32)
    obs, r, terminated, truncated, info = env.step(action)
    assert len(robot.sent_actions) == sent_before + 1
    assert r == 0.0 and terminated is False and truncated is False
    last = robot.sent_actions[-1]
    # 中位动作反归一化后 gripper≈50
    assert abs(last["gripper.pos"] - 50.0) < 1.0, last["gripper.pos"]

    # step: 边界动作 +1 应映射到 joint_upper
    action_max = np.ones(7, dtype=np.float32)
    env.step(action_max)
    last = robot.sent_actions[-1]
    assert abs(last["joint_1.pos"] - DEFAULT_JOINT_UPPER[0]) < 1.0
    assert abs(last["gripper.pos"] - DEFAULT_JOINT_UPPER[6]) < 1.0

    env.close()
    print("A10RobotEnv 基础测试通过")


from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.teleoperators.utils import TeleopEvents

if __name__ == "__main__":
    test_env_basic()
