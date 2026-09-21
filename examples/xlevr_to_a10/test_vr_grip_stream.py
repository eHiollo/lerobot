#!/usr/bin/env python3
"""Monitor the raw XLeVR side-grip stream without connecting to a robot.

From the LeRobot repository root:
    python examples/xlevr_to_a10/test_vr_grip_stream.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Keep the documented one-line command working even without an editable install.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

XLEVR_PATH = "/home/robot/VLA/XLeRobot/XLeVR"
DEFAULT_LOG_DIR = "outputs/vr_grip_tests"

RED = "\033[1;37;41m"
YELLOW = "\033[1;33m"
GREEN = "\033[1;32m"
RESET = "\033[0m"


def _plain_or_colored(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if sys.stdout.isatty() else text


def _json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class StreamSummary:
    polls: int = 0
    received_samples: int = 0
    duplicate_polls: int = 0
    no_data_polls: int = 0
    receive_gaps: int = 0
    websocket_disconnects: int = 0
    button_presses: int = 0
    button_off_edges: int = 0
    possible_button_bounces: int = 0
    max_receive_gap_ms: float = 0.0


class GripStreamMonitor:
    def __init__(self, gap_warning_s: float, bounce_window_s: float):
        self.gap_warning_s = gap_warning_s
        self.bounce_window_s = bounce_window_s
        self.summary = StreamSummary()
        self.last_sequence: int | None = None
        self.last_receive_time_s: float | None = None
        self.last_new_sample_local_s: float | None = None
        self.last_pressed = False
        self.last_off_local_s: float | None = None
        self.last_ws_clients: int | None = None
        self.gap_announced = False

    def update(self, action: dict[str, Any], status: dict[str, Any], now_s: float) -> dict[str, Any]:
        self.summary.polls += 1
        sequence = int(action.get("xlevr.sample_sequence", -1))
        receive_time_s = float(action.get("xlevr.sample_receive_time_s", -1.0))
        buttons = action.get("xlevr.buttons", {}) or {}
        grip_active = bool(action.get("xlevr.grip_active", False))
        squeeze = bool(buttons.get("squeeze", False))
        pressed = grip_active or squeeze
        ws_clients = int(status.get("ws_clients", 0) or 0)
        new_sample = sequence >= 0 and sequence != self.last_sequence
        events: list[str] = []
        receive_interval_s: float | None = None
        sequence_delta: int | None = None

        if new_sample:
            self.summary.received_samples += 1
            if self.last_sequence is not None:
                sequence_delta = sequence - self.last_sequence
            if self.last_receive_time_s is not None and receive_time_s >= 0.0:
                receive_interval_s = receive_time_s - self.last_receive_time_s
                if receive_interval_s >= 0.0:
                    self.summary.max_receive_gap_ms = max(
                        self.summary.max_receive_gap_ms, receive_interval_s * 1000.0
                    )
                if receive_interval_s > self.gap_warning_s and not self.gap_announced:
                    self.summary.receive_gaps += 1
                    events.append("receive_gap")
            if self.gap_announced:
                events.append("receive_recovered")
            self.last_sequence = sequence
            self.last_receive_time_s = receive_time_s
            self.last_new_sample_local_s = now_s
            self.gap_announced = False
        elif sequence < 0:
            self.summary.no_data_polls += 1
        else:
            self.summary.duplicate_polls += 1

        data_silence_s = (
            None if self.last_new_sample_local_s is None else now_s - self.last_new_sample_local_s
        )
        if (
            pressed
            and data_silence_s is not None
            and data_silence_s > self.gap_warning_s
            and not self.gap_announced
        ):
            self.summary.receive_gaps += 1
            events.append("receive_silence_while_pressed")
            self.gap_announced = True

        if pressed != self.last_pressed:
            if pressed:
                self.summary.button_presses += 1
                events.append("button_pressed")
                if (
                    self.last_off_local_s is not None
                    and now_s - self.last_off_local_s <= self.bounce_window_s
                ):
                    self.summary.possible_button_bounces += 1
                    events.append("possible_button_bounce")
            else:
                self.summary.button_off_edges += 1
                self.last_off_local_s = now_s
                events.append("button_off")
            self.last_pressed = pressed

        if self.last_ws_clients is not None and self.last_ws_clients > 0 and ws_clients == 0:
            self.summary.websocket_disconnects += 1
            events.append("websocket_disconnected")
        elif self.last_ws_clients == 0 and ws_clients > 0:
            events.append("websocket_connected")
        self.last_ws_clients = ws_clients

        return {
            "new_sample": new_sample,
            "sequence": sequence,
            "sequence_delta": sequence_delta,
            "receive_interval_s": receive_interval_s,
            "data_silence_s": data_silence_s,
            "pressed": pressed,
            "grip_active": grip_active,
            "squeeze": squeeze,
            "squeeze_value": buttons.get("squeeze_value"),
            "ws_clients": ws_clients,
            "events": events,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test continuous XLeVR side-grip data reception")
    parser.add_argument("--xlevr-path", default=XLEVR_PATH)
    parser.add_argument("--arm", default="right", choices=("left", "right"))
    parser.add_argument("--poll-hz", type=float, default=100.0)
    parser.add_argument(
        "--gap-warning-ms",
        type=float,
        default=150.0,
        help="按住侧键时超过该时间没有新 pose 就醒目报警",
    )
    parser.add_argument(
        "--bounce-window-ms",
        type=float,
        default=500.0,
        help="侧键关闭后在该时间内恢复，标为疑似抖动",
    )
    parser.add_argument("--status-interval", type=float, default=1.0)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.poll_hz <= 0 or args.gap_warning_ms <= 0 or args.bounce_window_ms < 0:
        raise ValueError("poll-hz/gap-warning-ms must be positive; bounce-window-ms cannot be negative")

    if args.log_file:
        log_path = Path(args.log_file).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        log_path = (Path(args.log_dir) / f"vr_grip_stream_{stamp}.jsonl").resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from lerobot.teleoperators.xlevr.config_xlevr import XLeVRTeleopConfig
        from lerobot.teleoperators.xlevr.teleop_xlevr import XLeVRTeleop
    except ModuleNotFoundError as exc:
        print(
            f"缺少 LeRobot 运行依赖 ({exc.name})。请先激活平时录制使用的 lerobot 环境。",
            flush=True,
        )
        return 2

    config = XLeVRTeleopConfig(
        xlevr_path=args.xlevr_path,
        arm=args.arm,
        control_fps=max(1, round(args.poll_hz)),
        enable_left_events=False,
    )
    teleop = XLeVRTeleop(config)
    monitor = GripStreamMonitor(args.gap_warning_ms / 1000.0, args.bounce_window_ms / 1000.0)
    period_s = 1.0 / args.poll_hz
    started_s = time.monotonic()
    last_status_print_s = -math.inf
    connected = False

    print("XLeVR 侧键连续数据测试（不连接机器人）", flush=True)
    print(f"日志: {log_path}", flush=True)
    print("进入 VR 后按住侧键；正常时每个新样本打印 [RX]，异常行以 !!! 标记。", flush=True)
    print("Ctrl+C 结束并打印统计。", flush=True)

    try:
        teleop.connect()
        connected = True
        with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
            while True:
                loop_started_s = time.monotonic()
                action = teleop.get_action()
                status = teleop.get_status()
                result = monitor.update(action, status, loop_started_s)
                record = {
                    "type": "poll",
                    "wall_time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                    "elapsed_s": loop_started_s - started_s,
                    "sample": result,
                    "action": _json_value(action),
                    "status": _json_value(status),
                }
                log_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

                for event in result["events"]:
                    if event in {
                        "receive_gap",
                        "receive_silence_while_pressed",
                        "websocket_disconnected",
                        "button_off",
                        "possible_button_bounce",
                    }:
                        interval_ms = (result["receive_interval_s"] or result["data_silence_s"] or 0) * 1000
                        message = (
                            f"!!! {event.upper()} !!! seq={result['sequence']} "
                            f"gap={interval_ms:.1f}ms grip={int(result['pressed'])} "
                            f"value={result['squeeze_value']} ws={result['ws_clients']}"
                        )
                        print(_plain_or_colored(message, RED), flush=True)
                    elif event == "receive_recovered":
                        print(
                            _plain_or_colored(
                                f"!!! RECEIVE RECOVERED !!! seq={result['sequence']}", GREEN
                            ),
                            flush=True,
                        )
                    elif event == "button_pressed":
                        print(
                            _plain_or_colored(
                                f">>> SIDE BUTTON ON <<< seq={result['sequence']} "
                                f"value={result['squeeze_value']}",
                                GREEN,
                            ),
                            flush=True,
                        )
                    elif event == "websocket_connected":
                        print(_plain_or_colored(">>> VR WEBSOCKET CONNECTED <<<", GREEN), flush=True)

                if result["pressed"] and result["new_sample"]:
                    interval_ms = (
                        result["receive_interval_s"] * 1000.0
                        if result["receive_interval_s"] is not None
                        else 0.0
                    )
                    age_ms = max(0.0, float(action.get("xlevr.sample_age_s", 0.0))) * 1000.0
                    print(
                        f"[RX] seq={result['sequence']:>7} dt={interval_ms:>6.1f}ms "
                        f"age={age_ms:>6.1f}ms grip=1 value={result['squeeze_value']} "
                        f"ws={result['ws_clients']}",
                        flush=True,
                    )
                elif loop_started_s - last_status_print_s >= args.status_interval:
                    color = GREEN if status.get("phase") == "receiving" else YELLOW
                    print(
                        _plain_or_colored(
                            f"[STATUS] phase={status.get('phase')} ws={result['ws_clients']} "
                            f"seq={result['sequence']} grip={int(result['pressed'])} "
                            f"age={status.get('last_pose_age_s')}",
                            color,
                        ),
                        flush=True,
                    )
                    last_status_print_s = loop_started_s

                time.sleep(max(0.0, period_s - (time.monotonic() - loop_started_s)))
    except KeyboardInterrupt:
        print("\n测试结束。", flush=True)
    except Exception as exc:
        print(_plain_or_colored(f"!!! TEST FAILED: {exc} !!!", RED), flush=True)
        return 1
    finally:
        finished_s = time.monotonic()
        summary = asdict(monitor.summary) | {"duration_s": finished_s - started_s}
        try:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(
                    json.dumps(
                        {
                            "type": "summary",
                            "wall_time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                            **summary,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        except OSError as exc:
            print(f"写入汇总失败: {exc}", flush=True)
        if connected:
            teleop.disconnect()
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"完整日志: {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
