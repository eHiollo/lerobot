# a10_tcp_bus.py
from __future__ import annotations

import json
import socket
import threading
from typing import Dict, Iterable, List

import cv2
import numpy as np


class A10TCPClient:
    """
    A10 机器人用的 TCP “总线”，API 风格参考 FeetechMotorsBus，但实现简单很多。

    - 一次性读/写整个关节向量，由对端控制器负责具体舵机通信
    - 只支持位置模式：Present_Position / Goal_Position
    - 同时可以拿到一帧 RGB 图像（用于数据集）
    
    [Singleton Mode]
    为了解决服务端只支持单客户端连接的问题，我们在客户端实现单例模式。
    如果 host 和 port 相同，则复用同一个连接。
    """
    
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, host: str, port: int, *args, **kwargs):
        key = (host, port)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = super().__new__(cls)
            return cls._instances[key]

    def __init__(
        self,
        host: str,
        port: int,
        joint_names: List[str] | None = None,
        timeout_ms: int = 3000,
    ) -> None:
        # 防止单例被多次初始化
        with self._lock:
            if hasattr(self, 'initialized') and self.initialized:
                # 如果新的 timeout 更长，更新它
                if timeout_ms > self.timeout_ms:
                    #print(f"[A10TCPClient] Updating timeout from {self.timeout_ms} to {timeout_ms} ms")
                    self.timeout_ms = timeout_ms
                    if self.sock:
                        self.sock.settimeout(self.timeout_ms / 1000.0)
                return
            
            self.host = host
            self.port = port
            self.timeout_ms = timeout_ms
            self.joint_names = joint_names or []

            self.sock: socket.socket | None = None
            self._last_q: np.ndarray | None = None
            self._buffer = b""
            
            # 事务锁，防止多线程（Leader/Follower）同时读写 socket 导致数据混乱
            self.tx_lock = threading.Lock()
            
            self.initialized = True

    # ---------- 连接 ----------

    @property
    def is_connected(self) -> bool:
        return self.sock is not None

    def connect(self, handshake: bool = True) -> None:
        """类似 FeetechMotorsBus.connect，只是底层是 TCP."""
        with self.tx_lock:
            if self.sock is not None:
                return

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            timeout_s = self.timeout_ms / 1000.0
            s.settimeout(timeout_s)
            try:
                s.connect((self.host, self.port))
            except socket.timeout as exc:
                s.close()
                raise ConnectionError(
                    f"连接 A10 控制器超时 ({timeout_s:.1f}s): {self.host}:{self.port}。"
                    f"请确认机器人 TCP 服务已启动且 IP/端口正确。"
                ) from exc
            except OSError as exc:
                s.close()
                raise ConnectionError(
                    f"无法连接 A10 控制器 {self.host}:{self.port}: {exc}。"
                    f"请确认机器人 TCP 服务已启动且网络可达。"
                ) from exc
            self.sock = s

            if handshake:
                self._handshake()

    def _handshake(self) -> None:
        """
        可选握手：比如向服务端发个 PING，看是否在线。
        不强制要求服务端实现；你可以按需修改。
        """
        try:
            # 如果你在服务端实现了 PING 命令，可以打开下面两行
            # self._send_line("PING")
            # resp = self._recvline()
            # assert resp == "PONG", f"Unexpected handshake reply: {resp}"
            pass
        except Exception as e:
            self.disconnect()
            raise ConnectionError(f"Handshake with A10 controller failed: {e!r}") from e

    def disconnect(self) -> None:
        with self.tx_lock:
            if self.sock is not None:
                try:
                    self.sock.close()
                finally:
                    self.sock = None
                    # 从单例池中移除？
                    # 考虑到可能还有其他引用，这里只关闭 socket。
                    # 如果需要彻底重置，可能需要更复杂的逻辑。

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
            print(f"[A10TCPClient] Timeout! Buffer content (hex): {self._buffer.hex()}")
            print(f"[A10TCPClient] Timeout! Buffer content (str): {self._buffer.decode('utf-8', errors='replace')}")
            raise TimeoutError("Socket timed out while reading chunk")

    def _recvline(self) -> str:
        assert self.sock is not None
        while b"\n" not in self._buffer:
            self._read_chunk()
        
        line_bytes, self._buffer = self._buffer.split(b"\n", 1)
        return line_bytes.decode("utf-8").strip()

    def _recv_exact(self, n: int) -> bytes:
        assert self.sock is not None
        while len(self._buffer) < n:
            self._read_chunk()
        
        data = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return data

    # ---------- 高层功能：状态 ----------

    def get_observation(self) -> dict:
        """
        获取follower关节信息。
        """
        with self.tx_lock:
            if not self.is_connected:
                raise ConnectionError("A10TCPBus is not connected")

            self._send_line("GET_FOLLOWER_STATE")
            
            # Loop to skip potential echoes or non-JSON lines
            while True:
                header_line = self._recvline()
                if not header_line:
                    raise ConnectionError("Received empty line from server")
                
                # Debug print: Show exactly what we received
                ##print(f"[A10TCPClient] Received (Follower): {header_line}")

                # If the line looks like JSON (starts with {), try to parse it
                if header_line.strip().startswith("{"):
                    try:
                        header = json.loads(header_line)
                        break
                    except json.JSONDecodeError:
                        print(f"[A10TCPClient] Warning: Failed to decode JSON line: {header_line}")
                        continue
                else:
                    # Ignore echoes like "SET_JOINTS ..." or "GET_STATE"
                    # print(f"[A10TCPClient] Debug: Ignored non-JSON line: {header_line}")
                    continue

            q = np.asarray(header["q"], dtype=np.float32)

            self._last_q = q
            return {"q": q}

    def get_state(self) -> dict:
        """
        获取leader关节信息。
        """
        with self.tx_lock:
            if not self.is_connected:
                raise ConnectionError("A10TCPBus is not connected")

            self._send_line("GET_LEADER_STATE")
            
            # Loop to skip potential echoes or non-JSON lines
            while True:
                header_line = self._recvline()
                if not header_line:
                    raise ConnectionError("Received empty line from server")
                
                # Debug print: Show exactly what we received
                #print(f"[A10TCPClient] Received (Leader): {header_line}")

                # If the line looks like JSON (starts with {), try to parse it
                if header_line.strip().startswith("{"):
                    try:
                        header = json.loads(header_line)
                        break
                    except json.JSONDecodeError:
                        print(f"[A10TCPClient] Warning: Failed to decode JSON line: {header_line}")
                        continue
                else:
                    # Ignore echoes like "SET_JOINTS ..." or "GET_STATE"
                    # print(f"[A10TCPClient] Debug: Ignored non-JSON line: {header_line}")
                    continue

            q = np.asarray(header["q"], dtype=np.float32)

            self._last_q = q
            return {"q": q}

    def send_action(self, q_target: np.ndarray) -> None:
        """
        发送目标关节位置。
        """
        with self.tx_lock:
            if not self.is_connected:
                raise ConnectionError("A10TCPBus is not connected")

            payload = json.dumps({"q": q_target.tolist()})
            cmd = f"SET_JOINTS {payload}"
            self._send_line(cmd)

            self._last_q = q_target

    def send_ee_delta(self, actions: list[float]) -> None:
        """
        Send 7D end-effector delta action at control rate (always).

        actions layout: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        When teleop is disabled, arm deltas (first 6) are zero; gripper is still sent.
        """
        if len(actions) != 7:
            raise ValueError(f"SET_EE_DELTA expects 7 actions, got {len(actions)}")

        with self.tx_lock:
            if not self.is_connected:
                raise ConnectionError("A10TCPBus is not connected")

            payload = json.dumps({"actions": [float(v) for v in actions]})
            cmd = f"SET_EE_DELTA {payload}"
            self._send_line(cmd)


    # ---------- Feetech 风格 API：read / sync_read ----------

    def sync_read(
        self,
        data_name: str,
        motors: Iterable[str] | None = None,
    ) -> Dict[str, float]:
        """
        类似 FeetechMotorsBus.sync_read：
        - 这里只支持 data_name == 'Present_Position'
        - 一次性从控制器读出整组关节，再按名字打包成 dict
        """
        if data_name != "Present_Position":
            raise ValueError(
                f"A10TCPBus only supports sync_read('Present_Position'), got {data_name!r}"
            )

        # 注意：sync_read 通常用于 Follower 机器人读取自己的状态
        # 所以这里应该调用 get_observation (GET_FOLLOWER_STATE)
        state = self.get_observation()
        q = state["q"]

        if motors is None:
            motors = self.joint_names

        motors = list(motors)
        if len(motors) != len(q):
            # 简单假设顺序对齐：joint_1 -> q[0], ...
            # 如果未来有子集，可以放宽逻辑
            if len(q) < len(motors):
                raise ValueError(
                    f"Controller returned {len(q)} joints, but motors list has {len(motors)} elements"
                )

        pos_dict: Dict[str, float] = {}
        for i, name in enumerate(motors):
            pos_dict[name] = float(q[i])

        return pos_dict

    def read(self, data_name: str, motor: str) -> float:
        """
        单个关节读值的捷径，行为类似 MotorsBus.read。
        """
        vals = self.sync_read(data_name, motors=[motor])
        return vals[motor]

    # ---------- Feetech 风格 API：write / sync_write ----------

    def sync_write(
        self,
        data_name: str,
        values: Dict[str, float],
    ) -> None:
        """
        类似 FeetechMotorsBus.sync_write：
        - 这里只支持 data_name == 'Goal_Position'
        - 一次性发送整组关节目标
        """
        if data_name != "Goal_Position":
            raise ValueError(
                f"A10TCPBus only supports sync_write('Goal_Position', ...), got {data_name!r}"
            )

        if not self.is_connected:
            raise ConnectionError("A10TCPBus is not connected")

        # 先拿一个基准向量：优先用上次的 q，否则 0
        if self._last_q is not None:
            q_target = self._last_q.copy()
        else:
            q_target = np.zeros(len(self.joint_names), dtype=np.float32)

        name_to_idx = {name: i for i, name in enumerate(self.joint_names)}

        for name, val in values.items():
            if name not in name_to_idx:
                raise KeyError(f"Unknown joint name {name!r}")
            q_target[name_to_idx[name]] = float(val)

        # 通过底层 SET_JOINTS 协议发送
        payload = json.dumps({"q": q_target.tolist()})
        cmd = f"SET_JOINTS {payload}"
        #print(f"[A10TCPClient] Sending: {cmd}")
        self._send_line(cmd)

        # 你可以在服务端回一个 OK，这里可选读一行响应：
        # resp = self._recvline()
        # if resp != "OK":
        #     raise RuntimeError(f"Unexpected reply to SET_JOINTS: {resp!r}")

        self._last_q = q_target

    def write(self, data_name: str, motor: str, value: float) -> None:
        """
        单个关节写值的捷径，行为类似 MotorsBus.write。
        """
        self.sync_write(data_name, {motor: value})

    # ---------- 可选：扭矩开关 ----------

    def enable_torque(self) -> None:
        """
        如果未来你在服务端实现 ENABLE_TORQUE 命令，可以在这里封装。
        现在先写成占位，不强制使用。
        """
        if not self.is_connected:
            raise ConnectionError("A10TCPBus is not connected")
        # self._send_line("ENABLE_TORQUE")

    def disable_torque(self) -> None:
        """
        同上，可选实现 DISABLE_TORQUE。
        """
        if not self.is_connected:
            raise ConnectionError("A10TCPBus is not connected")
        # self._send_line("DISABLE_TORQUE")
