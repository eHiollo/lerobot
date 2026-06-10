"""
Bridge to XLeVR's VRMonitor with a configurable project path.
"""

from __future__ import annotations

import asyncio
import http.server
import logging
import os
import socket
import ssl
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """检测 TCP 端口是否已被占用。"""
    check_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((check_host, port))
            return False
        except OSError:
            return True


def format_port_busy_help(https_port: int, ws_port: int | None = None) -> str:
    ws_line = f"\n  WebSocket 端口 {ws_port} 也可能被占用。" if ws_port is not None else ""
    return (
        f"XLeVR 端口 {https_port} 已被占用（Address already in use）。\n"
        f"通常是因为之前的 teleoperate 仍在运行。{ws_line}\n"
        f"处理：\n"
        f"  1) 在旧终端 Ctrl+C 结束进程，或\n"
        f"  2) 查占用: ss -tlnp | grep -E '{https_port}|{ws_port or ''}'\n"
        f"  3) 结束进程: kill <PID>\n"
        f"然后只保留一个 XLeVR 实例再启动（VR 浏览器可继续用原 https 页面）。"
    )


class ReuseHTTPServer(http.server.HTTPServer):
    allow_reuse_address = True


def setup_xlevr_environment(xlevr_path: str) -> None:
    xlevr_path = str(Path(xlevr_path).resolve())
    if xlevr_path not in sys.path:
        sys.path.insert(0, xlevr_path)
    os.environ["PYTHONPATH"] = f"{xlevr_path}:{os.environ.get('PYTHONPATH', '')}"


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "localhost"


def import_xlevr_modules():
    try:
        from xlevr.config import XLeVRConfig
        from xlevr.inputs.base import ControlGoal, ControlMode
        from xlevr.inputs.vr_ws_server import VRWebSocketServer

        return XLeVRConfig, VRWebSocketServer, ControlGoal, ControlMode
    except ImportError as exc:
        logger.error("Failed to import XLeVR modules: %s", exc)
        return None, None, None, None


class SimpleAPIHandler(http.server.BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        try:
            super().end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, ssl.SSLError):
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, _format, *_args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.serve_file("web-ui/index.html", "text/html")
        elif self.path.endswith(".css"):
            self.serve_file(f"web-ui{self.path}", "text/css")
        elif self.path.endswith(".js"):
            self.serve_file(f"web-ui{self.path}", "application/javascript")
        elif self.path.endswith(".ico"):
            self.serve_file(self.path[1:], "image/x-icon")
        elif self.path.endswith((".jpg", ".jpeg", ".png", ".gif")):
            content_type = (
                "image/jpeg"
                if self.path.endswith((".jpg", ".jpeg"))
                else "image/png"
                if self.path.endswith(".png")
                else "image/gif"
            )
            self.serve_file(f"web-ui{self.path}", content_type)
        else:
            self.send_error(404, "Not found")

    def serve_file(self, filename: str, content_type: str):
        try:
            web_root = getattr(self.server, "web_root_path", ".")
            file_path = os.path.join(web_root, filename)
            if os.path.exists(file_path):
                with open(file_path, "rb") as file_obj:
                    content = file_obj.read()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, f"File not found: {filename}")
        except Exception as exc:
            logger.error("Error serving %s: %s", filename, exc)
            self.send_error(500, "Internal server error")


class SimpleHTTPSServer:
    def __init__(self, config, web_root_path: str):
        self.config = config
        self.httpd = None
        self.server_thread = None
        self.web_root_path = web_root_path

    async def start(self):
        self.httpd = ReuseHTTPServer((self.config.host_ip, self.config.https_port), SimpleAPIHandler)
        self.httpd.web_root_path = self.web_root_path

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        certfile = os.path.join(self.web_root_path, self.config.certfile)
        keyfile = os.path.join(self.web_root_path, self.config.keyfile)
        context.load_cert_chain(certfile, keyfile)
        self.httpd.socket = context.wrap_socket(self.httpd.socket, server_side=True)

        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()
        logger.info(
            "XLeVR HTTPS server started on %s:%s", self.config.host_ip, self.config.https_port
        )

    async def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            if self.server_thread:
                self.server_thread.join(timeout=5)


class VRMonitorBridge:
    """Thread-safe wrapper around XLeVR VR WebSocket + HTTPS servers."""

    def __init__(self, xlevr_path: str):
        self.xlevr_path = str(Path(xlevr_path).resolve())
        self.config = None
        self.vr_server = None
        self.https_server = None
        self.is_running = False
        self.latest_goal = None
        self.left_goal = None
        self.right_goal = None
        self.headset_goal = None
        self._goal_lock = threading.Lock()
        self.command_queue = None
        self.goals_received = 0
        self.last_goal_time: float | None = None
        self.servers_started = False
        self.startup_error: BaseException | None = None

    def initialize(self) -> bool:
        if not Path(self.xlevr_path).exists():
            logger.error("XLeVR path does not exist: %s", self.xlevr_path)
            return False

        setup_xlevr_environment(self.xlevr_path)
        original_cwd = os.getcwd()
        os.chdir(self.xlevr_path)

        try:
            XLeVRConfig, VRWebSocketServer, _, _ = import_xlevr_modules()
            if XLeVRConfig is None:
                return False

            self.config = XLeVRConfig()
            self.config.enable_vr = True
            self.config.enable_keyboard = False
            self.config.enable_https = True

            self.command_queue = asyncio.Queue()
            self.vr_server = VRWebSocketServer(
                command_queue=self.command_queue,
                config=self.config,
                print_only=False,
            )
            self.https_server = SimpleHTTPSServer(self.config, self.xlevr_path)
            return True
        except Exception as exc:
            logger.error("Failed to initialize XLeVR monitor: %s", exc)
            return False
        finally:
            os.chdir(original_cwd)

    async def start_monitoring(self):
        if self.config is None or self.vr_server is None:
            if not self.initialize():
                return

        original_cwd = os.getcwd()
        os.chdir(self.xlevr_path)
        try:
            try:
                await self.https_server.start()
                await self.vr_server.start()
            except OSError as exc:
                self.startup_error = exc
                logger.error("XLeVR server bind failed: %s", exc)
                return

            self.is_running = True
            self.servers_started = True

            host_display = get_local_ip() if self.config.host_ip == "0.0.0.0" else self.config.host_ip
            ws_port = self.config.websocket_port
            print(
                f"[XLeVR] 服务已启动\n"
                f"  网页: https://{host_display}:{self.config.https_port}\n"
                f"  WebSocket: wss://{host_display}:{ws_port}\n"
                f"  请在 VR 浏览器打开上面的 https 地址，并点击 Enter VR / Start",
                flush=True,
            )
            logger.info("Open VR browser at https://%s:%s", host_display, self.config.https_port)
            await self._monitor_commands()
        finally:
            await self.stop_monitoring()
            os.chdir(original_cwd)

    async def _monitor_commands(self):
        while self.is_running:
            try:
                goal = await asyncio.wait_for(self.command_queue.get(), timeout=1.0)
                with self._goal_lock:
                    if goal.arm == "left":
                        self.left_goal = goal
                    elif goal.arm == "right":
                        self.right_goal = self._merge_goal(self.right_goal, goal)
                    elif goal.arm == "headset":
                        self.headset_goal = goal
                    self.latest_goal = goal
                self.goals_received += 1
                self.last_goal_time = time.time()
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.error("Error processing VR command: %s", exc)

    @staticmethod
    def _merge_goal(previous: Any, current: Any) -> Any:
        """Keep last pose / input state when a partial goal arrives."""
        if previous is None:
            return current
        if current.target_position is None and previous.target_position is not None:
            current.target_position = previous.target_position
        prev_meta = getattr(previous, "metadata", None) or {}
        cur_meta = getattr(current, "metadata", None) or {}
        merged_meta = dict(prev_meta)
        merged_meta.update(cur_meta)
        if merged_meta.get("orientation_quat") is None and prev_meta.get("orientation_quat") is not None:
            merged_meta["orientation_quat"] = prev_meta["orientation_quat"]

        # One-shot flags must not stick across later position updates.
        if not cur_meta.get("reset_target_to_current"):
            merged_meta.pop("reset_target_to_current", None)

        if not cur_meta.get("buttons") and prev_meta.get("buttons"):
            merged_meta["buttons"] = prev_meta["buttons"]
        if "trigger" not in cur_meta and "trigger" in prev_meta:
            merged_meta["trigger"] = prev_meta["trigger"]
        if "grip_active" not in cur_meta and "grip_active" in prev_meta:
            merged_meta["grip_active"] = prev_meta["grip_active"]
        current.metadata = merged_meta
        return current

    def get_latest_goal_nowait(self, arm: str | None = None) -> Any:
        with self._goal_lock:
            if arm == "left":
                return self.left_goal
            if arm == "right":
                return self.right_goal
            if arm == "headset":
                return self.headset_goal
            return {
                "left": self.left_goal,
                "right": self.right_goal,
                "headset": self.headset_goal,
                "has_left": self.left_goal is not None,
                "has_right": self.right_goal is not None,
                "has_headset": self.headset_goal is not None,
            }

    def get_status(self) -> dict[str, Any]:
        """Return VR server / client connectivity snapshot."""
        ws_clients = 0
        if self.vr_server is not None and hasattr(self.vr_server, "clients"):
            ws_clients = len(self.vr_server.clients)

        now = time.time()
        last_age = None if self.last_goal_time is None else now - self.last_goal_time

        if ws_clients == 0:
            phase = "waiting_browser"
            hint = "VR 浏览器还没连上 WebSocket，请打开 https 页面并进入 VR"
        elif self.goals_received == 0:
            phase = "browser_connected_no_data"
            hint = "浏览器已连接，但还没收到手柄数据，请在 VR 里移动手柄"
        elif last_age is not None and last_age > 3.0:
            phase = "stale"
            hint = f"超过 {last_age:.1f}s 没收到新数据，检查 VR 页面是否仍在前台"
        else:
            phase = "receiving"
            hint = "正在接收 VR 数据"

        cert_ok = False
        if self.config is not None:
            cert_path = os.path.join(self.xlevr_path, self.config.certfile)
            key_path = os.path.join(self.xlevr_path, self.config.keyfile)
            cert_ok = os.path.exists(cert_path) and os.path.exists(key_path)

        return {
            "servers_started": self.servers_started,
            "is_running": self.is_running,
            "thread_alive": self.is_running,
            "ssl_cert_ok": cert_ok,
            "ws_clients": ws_clients,
            "goals_received": self.goals_received,
            "last_goal_age_s": last_age,
            "has_left_goal": self.left_goal is not None,
            "has_right_goal": self.right_goal is not None,
            "phase": phase,
            "hint": hint,
            "https_port": getattr(self.config, "https_port", None),
            "websocket_port": getattr(self.config, "websocket_port", None),
        }

    async def stop_monitoring(self):
        self.is_running = False
        if self.vr_server:
            await self.vr_server.stop()
        if self.https_server:
            await self.https_server.stop()
