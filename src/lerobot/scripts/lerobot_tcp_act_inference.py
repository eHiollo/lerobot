#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. Licensed under the Apache License, Version 2.0.

"""
TCP **server** (bind on this machine). The simulator connects **as a client** — same direction as a typical
OpenPI / policy server where IsaacLab pushes observations and receives joint targets.

Recommended frame protocol (default, ``--protocol push_json``): **one JSON object per control step**, one line,
ending with ``\\n``:

  - ``follower_q`` / ``robot_q`` / ``q_obs``: length-7 list (or nested, see below).
  - ``wrist_image_jpeg_b64`` / ``wrist_jpeg_b64``: base64 JPEG (RGB wrist / egocentric).
  - ``top_image_jpeg_b64`` / ``top_jpeg_b64`` / ``base_image_jpeg_b64``: base64 JPEG (fixed / third-person).
    If top is omitted, wrist is duplicated for ``observation.images.top`` (domain gap vs training).

Server replies with one line::

  {"q": [t0, ..., t6]}\\n

Optional compatibility mode ``--protocol hybrid``: after each push_json line, caches telemetry and still answers
``GET_LEADER_STATE``, ``GET_FOLLOWER_STATE``, ``GET_WRIST_RGB``, ``GET_TOP_RGB`` **exactly like** the simulator-side
``A10TcpServer`` (same JSON shapes). Use this only if your **unchanged** client already sends GET_* after filling
state via an observation push.

Example::

python -m lerobot.scripts.lerobot_tcp_act_inference \\
  --policy.path=/home/me/lerobot/outputs/run/checkpoints/004000/pretrained_model \\
  --bind-host=0.0.0.0 \\
  --bind-port=8766 \\
  --dataset.repo_id=dat_5_9 \\
  --dataset.root=/home/me/lerobot/dataset/data_5_9

Bind host defaults to all interfaces so the simulator host can connect over the network.

Security: TLS is not used — keep this on a trusted LAN/VPN.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import socket
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device

logger = logging.getLogger(__name__)


def read_line(sock: socket.socket, buf: bytearray, max_line_bytes: int = 80 * 1024 * 1024) -> str | None:
    while True:
        pos = buf.find(b"\n")
        if pos != -1:
            if pos > max_line_bytes:
                raise ValueError(f"Line longer than {max_line_bytes} bytes")
            line = buf[:pos].decode("utf-8", errors="replace")
            del buf[: pos + 1]
            return line
        chunk = sock.recv(65536)
        if not chunk:
            return None
        buf.extend(chunk)


def send_all(sock: socket.socket, text: str) -> None:
    if not text.endswith("\n"):
        text = text + "\n"
    data = text.encode("utf-8")
    sock.sendall(data)


def decode_jpeg_b64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def resize_hwc_uint8(img_hwc: np.ndarray, height: int, width: int) -> np.ndarray:
    t = torch.from_numpy(img_hwc).permute(2, 0, 1).float() / 255.0
    t = TVF.resize(t, (height, width), antialias=True)
    out = (t * 255.0).clamp(0, 255).byte().permute(1, 2, 0).contiguous().numpy()
    return out


def _flatten_state(obj: Any) -> np.ndarray | None:
    """Extract length-7 float vector from nested / flat JSON."""
    if obj is None:
        return None
    if isinstance(obj, list):
        arr = np.asarray(obj, dtype=np.float32).reshape(-1)
        if arr.size >= 7:
            return arr[:7].astype(np.float32)
        return None
    if isinstance(obj, dict):
        for k in ("observation/state", "state"):
            if k in obj:
                return _flatten_state(obj[k])
        if "observation" in obj:
            return _flatten_state(obj["observation"])
    return None


def extract_observation_payload(obj: dict[str, Any]) -> tuple[np.ndarray, str, str] | None:
    """
    Returns (state7, wrist_b64, top_b64). Top may equal wrist if absent.
    Supports flat keys and nested ``observation`` dict similar to OpenPI-style payloads.
    """
    st = None
    for key in ("follower_q", "robot_q", "q_obs", "observation_state"):
        if key in obj:
            st = _flatten_state(obj[key])
            break
    if st is None and "observation" in obj and isinstance(obj["observation"], dict):
        obsd = obj["observation"]
        if "state" in obsd:
            st = _flatten_state(obsd["state"])
        if st is None and "observation" in obsd:
            st = _flatten_state(obsd["observation"])
    if st is None:
        st = _flatten_state(obj.get("observation/state"))

    wb64 = None
    for key in (
        "wrist_image_jpeg_b64",
        "wrist_jpeg_b64",
        "image_right_jpeg_b64",
        "right_jpeg_b64",
    ):
        if key in obj and isinstance(obj[key], str):
            wb64 = obj[key]
            break
    if wb64 is None and isinstance(obj.get("observation"), dict):
        o = obj["observation"]
        for key in ("images/right", "images/right_b64", "image_right", "right"):
            if key in o and isinstance(o[key], str):
                wb64 = o[key]
                break

    tb64 = None
    for key in (
        "top_image_jpeg_b64",
        "top_jpeg_b64",
        "base_image_jpeg_b64",
        "image_top_jpeg_b64",
    ):
        if key in obj and isinstance(obj[key], str):
            tb64 = obj[key]
            break
    if tb64 is None and isinstance(obj.get("observation"), dict):
        o = obj["observation"]
        for key in ("images/top", "images/base", "image_top", "top"):
            if key in o and isinstance(o[key], str):
                tb64 = o[key]
                break

    if st is None or wb64 is None:
        return None
    if tb64 is None:
        tb64 = wb64
    return st, wb64, tb64


class A10MirrorCache:
    """Caches robot_q + JPEG payloads so GET_* lines can be answered like ``A10TcpServer``."""

    def __init__(self) -> None:
        self.robot_q: list[float] = [0.0] * 13
        self.wrist_rgb_jpeg_b64: str = ""
        self.wrist_rgb_shape: list[int] = [0, 0, 0]
        self.wrist_rgb_ts: float = 0.0
        self.top_rgb_jpeg_b64: str = ""
        self.top_rgb_shape: list[int] = [0, 0, 0]
        self.top_rgb_ts: float = 0.0

    def update_from_observation(self, follower7: np.ndarray, wrist_hwc: np.ndarray, top_hwc: np.ndarray) -> None:
        q = list(self.robot_q)
        for i in range(min(7, follower7.shape[0])):
            q[i] = float(follower7[i])
        self.robot_q = q

        def pack(hwc: np.ndarray) -> tuple[str, list[int], float]:
            buf = io.BytesIO()
            Image.fromarray(hwc).save(buf, format="JPEG", quality=92)
            ts = time.time()
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            shp = [int(hwc.shape[0]), int(hwc.shape[1]), 3]
            return b64, shp, ts

        wb64, wsh, wts = pack(wrist_hwc)
        tb64, tsh, tts = pack(top_hwc)
        self.wrist_rgb_jpeg_b64 = wb64
        self.wrist_rgb_shape = wsh
        self.wrist_rgb_ts = wts
        self.top_rgb_jpeg_b64 = tb64
        self.top_rgb_shape = tsh
        self.top_rgb_ts = tts

    @staticmethod
    def _select_leader_vals(current_q: list[float]) -> list[float]:
        if len(current_q) >= 12:
            return [current_q[i + 6] for i in range(6)]
        vals = list(current_q[:6])
        while len(vals) < 6:
            vals.append(0.0)
        return vals

    @staticmethod
    def _select_follower_vals(current_q: list[float]) -> list[float]:
        if len(current_q) >= 13:
            vals = [current_q[i] for i in range(6)]
            vals.append(current_q[12])
            return vals
        if len(current_q) >= 7:
            vals = [current_q[i] for i in range(6)]
            vals.append(current_q[6])
            return vals
        vals = list(current_q[:7])
        while len(vals) < 7:
            vals.append(0.0)
        return vals

    def send_leader_state(self, sock: socket.socket) -> None:
        vals = self._select_leader_vals(self.robot_q)
        payload = "{\"q\": [" + ", ".join(str(v) for v in vals) + "]}\n"
        sock.sendall(payload.encode("utf-8"))

    def send_follower_state(self, sock: socket.socket) -> None:
        vals = self._select_follower_vals(self.robot_q)
        payload = "{\"q\": [" + ", ".join(str(v) for v in vals) + "]}\n"
        sock.sendall(payload.encode("utf-8"))

    def send_jpeg_line(self, sock: socket.socket, b64: str, shape: list[int], ts: float) -> None:
        payload = (
            "{"
            + f"\"image_jpeg_b64\": \"{b64}\", "
            + f"\"shape\": [{shape[0]}, {shape[1]}, {shape[2]}], "
            + f"\"ts\": {ts}"
            + "}\n"
        )
        sock.sendall(payload.encode("utf-8"))

    def handle_get_line(self, sock: socket.socket, line: str) -> bool:
        if "GET_LEADER_STATE" in line:
            self.send_leader_state(sock)
            return True
        if "GET_FOLLOWER_STATE" in line:
            self.send_follower_state(sock)
            return True
        if "GET_WRIST_RGB" in line:
            self.send_jpeg_line(sock, self.wrist_rgb_jpeg_b64, self.wrist_rgb_shape, self.wrist_rgb_ts)
            return True
        if "GET_TOP_RGB" in line:
            self.send_jpeg_line(sock, self.top_rgb_jpeg_b64, self.top_rgb_shape, self.top_rgb_ts)
            return True
        return False


def load_dataset_meta_from_train_config(pretrained_dir: Path) -> tuple[str, str] | None:
    tc = pretrained_dir / "train_config.json"
    if not tc.is_file():
        return None
    with open(tc, encoding="utf-8") as f:
        cfg = json.loads(f.read())
    ds = cfg.get("dataset") or {}
    rid = ds.get("repo_id")
    root = ds.get("root")
    if isinstance(rid, str) and isinstance(root, str):
        return rid, root
    return None


def build_numpy_obs(
    state7: np.ndarray,
    wrist_hwc: np.ndarray,
    top_hwc: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "observation.state": state7.astype(np.float32).reshape(7),
        "observation.images.right": wrist_hwc,
        "observation.images.top": top_hwc,
    }


def handle_client(
    conn: socket.socket,
    addr: tuple,
    *,
    policy,
    preprocessor,
    postprocessor,
    policy_cfg: PreTrainedConfig,
    device: torch.device,
    exp_h: int,
    exp_w: int,
    protocol: str,
    mirror: A10MirrorCache,
    infer_lock: threading.Lock,
) -> None:
    buf = bytearray()
    logger.info("Client connected: %s", addr)
    try:
        while True:
            line = read_line(conn, buf)
            if line is None:
                break
            line = line.strip()
            if not line:
                continue

            if mirror.handle_get_line(conn, line):
                continue

            if protocol == "get_only":
                logger.warning("Ignoring non-GET line in get_only mode: %s...", line[:120])
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Non-JSON line (ignored): %s...", line[:200])
                continue

            if not isinstance(obj, dict):
                continue

            extracted = extract_observation_payload(obj)
            if extracted is None:
                logger.warning("JSON missing follower_q / wrist jpeg keys; keys=%s", list(obj.keys())[:20])
                continue

            st7, wb64, tb64 = extracted
            wrist_hwc = decode_jpeg_b64(wb64)
            top_hwc = decode_jpeg_b64(tb64)
            wrist_r = resize_hwc_uint8(wrist_hwc, exp_h, exp_w)
            top_r = resize_hwc_uint8(top_hwc, exp_h, exp_w)

            mirror.update_from_observation(st7, wrist_hwc, top_hwc)

            observation = build_numpy_obs(st7, wrist_r, top_r)
            with infer_lock:
                action_t = predict_action(
                    observation=observation,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy_cfg.use_amp,
                    task=str(obj.get("prompt", "") or ""),
                    robot_type=None,
                )
            action_np = action_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
            cmd = json.dumps({"q": [float(x) for x in action_np[:7].tolist()]})
            send_all(conn, cmd)

    except ConnectionResetError:
        logger.info("Client reset %s", addr)
    except Exception:
        logger.exception("Client handler error %s", addr)
    finally:
        conn.close()
        logger.info("Client disconnected: %s", addr)


def main() -> None:
    parser = argparse.ArgumentParser(description="LeRobot ACT TCP policy server (simulator connects as client).")
    parser.add_argument("--policy.path", dest="policy_path", type=str, required=True)
    parser.add_argument("--bind-host", type=str, default="0.0.0.0", help="Listen address.")
    parser.add_argument("--bind-port", type=int, required=True)
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", type=str, default=None)
    parser.add_argument("--dataset.root", dest="dataset_root", type=str, default=None)
    parser.add_argument(
        "--protocol",
        type=str,
        choices=("push_json", "hybrid", "get_only"),
        default="push_json",
        help=(
            "push_json: each line is one observation JSON → reply action. "
            "hybrid: same + answer GET_* using last observation cache (A10TcpServer-compatible replies). "
            "get_only: only GET_* (you must fill cache another way — not recommended)."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    pretrained_path = Path(args.policy_path).expanduser().resolve()
    if not (pretrained_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json under {pretrained_path}")

    ds_from_train = load_dataset_meta_from_train_config(pretrained_path)
    repo_id = args.dataset_repo_id or (ds_from_train[0] if ds_from_train else None)
    root = args.dataset_root or (ds_from_train[1] if ds_from_train else None)
    if not repo_id or not root:
        raise ValueError("Provide --dataset.repo_id and --dataset.root or ship train_config.json beside checkpoint.")

    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    policy_cfg = PreTrainedConfig.from_pretrained(str(pretrained_path))
    policy_cfg.pretrained_path = pretrained_path
    device = get_safe_torch_device(policy_cfg.device, log=False)

    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(pretrained_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    right_ft = policy_cfg.input_features["observation.images.right"]
    _, exp_h, exp_w = right_ft.shape
    policy.reset()

    infer_lock = threading.Lock()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.bind_host, args.bind_port))
    listener.listen(8)
    logger.info("ACT TCP server listening on %s:%s (protocol=%s)", args.bind_host, args.bind_port, args.protocol)

    try:
        while True:
            conn, addr = listener.accept()
            conn.settimeout(600.0)
            mirror = A10MirrorCache()
            t = threading.Thread(
                target=handle_client,
                args=(conn, addr),
                kwargs={
                    "policy": policy,
                    "preprocessor": preprocessor,
                    "postprocessor": postprocessor,
                    "policy_cfg": policy_cfg,
                    "device": device,
                    "exp_h": exp_h,
                    "exp_w": exp_w,
                    "protocol": args.protocol,
                    "mirror": mirror,
                    "infer_lock": infer_lock,
                },
                daemon=True,
            )
            t.start()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        listener.close()


if __name__ == "__main__":
    main()
