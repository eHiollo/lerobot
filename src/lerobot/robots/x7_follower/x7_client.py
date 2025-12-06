# x7_client.py
from __future__ import annotations

import json
import socket
from typing import Dict, Iterable, List

import numpy as np


class X7TCPClient:
    """
    X7 机器人用的 TCP 客户端。
    只负责关节数据的收发 (8 DOF: 7 joints + 1 gripper)。
    图像数据由 LeRobot 的标准 Camera 类在本地处理。
    """

    def __init__(
        self,
        host: str,
        port: int,
        joint_names: List[str] | None = None,
        timeout_ms: int = 1000,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.joint_names = joint_names or []

        self.sock: socket.socket | None = None
        self._last_q: np.ndarray | None = None
        self._buffer = b""

    # ---------- 连接 ----------

    @property
    def is_connected(self) -> bool:
        return self.sock is not None

    def connect(self, handshake: bool = True) -> None:
        if self.sock is not None:
            return

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout_ms / 1000.0)
        s.connect((self.host, self.port))
        self.sock = s

        if handshake:
            self._handshake()

    def _handshake(self) -> None:
        try:
            # 可选握手
            pass
        except Exception as e:
            self.disconnect()
            raise ConnectionError(f"Handshake with X7 controller failed: {e!r}") from e

    def disconnect(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    # ---------- 底层收发工具 ----------

    def _send_line(self, line: str) -> None:
        assert self.sock is not None, "Socket is not connected"
        data = (line + "\n").encode("utf-8")
        self.sock.sendall(data)

    def _read_chunk(self) -> None:
        assert self.sock is not None
        try:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("TCP connection closed while reading")
            self._buffer += chunk
        except socket.timeout:
            raise TimeoutError("Socket timed out while reading chunk")

    def _recvline(self) -> str:
        assert self.sock is not None
        while b"\n" not in self._buffer:
            self._read_chunk()
        
        line_bytes, self._buffer = self._buffer.split(b"\n", 1)
        return line_bytes.decode("utf-8").strip()

    # ---------- 高层功能：状态 ----------

    def get_state(self) -> dict:
        """
        获取关节信息。
        """
        if not self.is_connected:
            raise ConnectionError("X7TCPClient is not connected")

        self._send_line("GET_STATE")
        header_line = self._recvline()
        header = json.loads(header_line)

        q = np.asarray(header["q"], dtype=np.float32)

        self._last_q = q
        return {"q": q}

    def send_action(self, q_target: np.ndarray) -> None:
        """
        发送目标关节位置。
        """
        if not self.is_connected:
            raise ConnectionError("X7TCPClient is not connected")

        payload = json.dumps({"q": q_target.tolist()})
        self._send_line(f"SET_JOINTS {payload}")
        
        self._last_q = q_target

    # ---------- Feetech 风格 API ----------

    def sync_read(
        self,
        data_name: str,
        motors: Iterable[str] | None = None,
    ) -> Dict[str, float]:
        if data_name != "Present_Position":
            raise ValueError(
                f"X7TCPClient only supports sync_read('Present_Position'), got {data_name!r}"
            )

        state = self.get_state()
        q = state["q"]

        if motors is None:
            motors = self.joint_names

        motors = list(motors)
        # 允许 q 的长度大于 motors (例如服务端返回更多数据)
        # 但如果 q 比 motors 短，则有问题
        if len(q) < len(motors):
             raise ValueError(
                f"Controller returned {len(q)} joints, but motors list has {len(motors)} elements"
            )

        pos_dict: Dict[str, float] = {}
        for i, name in enumerate(motors):
            pos_dict[name] = float(q[i])

        return pos_dict

    def sync_write(
        self,
        data_name: str,
        values: Dict[str, float],
    ) -> None:
        if data_name != "Goal_Position":
            raise ValueError(
                f"X7TCPClient only supports sync_write('Goal_Position', ...), got {data_name!r}"
            )

        if not self.is_connected:
            raise ConnectionError("X7TCPClient is not connected")

        if self._last_q is not None:
            q_target = self._last_q.copy()
        else:
            # 默认全0，长度为 joint_names 的长度
            q_target = np.zeros(len(self.joint_names), dtype=np.float32)

        name_to_idx = {name: i for i, name in enumerate(self.joint_names)}

        for name, val in values.items():
            if name not in name_to_idx:
                raise KeyError(f"Unknown joint name {name!r}")
            q_target[name_to_idx[name]] = float(val)

        payload = json.dumps({"q": q_target.tolist()})
        self._send_line(f"SET_JOINTS {payload}")

        self._last_q = q_target
