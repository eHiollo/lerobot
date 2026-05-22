#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. Licensed under the Apache License, Version 2.0.

"""
OpenPI-compatible WebSocket policy server for **LeRobot ACT** checkpoints.

Mirrors `openpi.serving.websocket_policy_server.WebsocketPolicyServer` so Isaac / ``openpi_client.WebsocketClientPolicy``
works unchanged: connect to ``ws://<host>:<port>``, receive msgpack **metadata** first, then send msgpack **observations**
and receive msgpack **actions** (with an ``actions`` array).

Install::

  pip install "lerobot[ws-act]"   # websockets + msgpack

Run::

  python -m lerobot.scripts.lerobot_ws_act_policy_server \\
    --policy.path=/path/to/pretrained_model \\
    --port=8000 \\
    --dataset.repo_id=... \\
    --dataset.root=...

Point IsaacLab at the same host/port as OpenPI (``--policy_host`` / ``--policy_port`` or env vars).

Observation mapping
---------------------
Your sim likely sends the same dict as for pi0.5 (``get_observation``): keys like ``observation/state``,
``observation/images/right``, ``observation/images/top`` with images **uint8 CHW** (224×224) and state ``(1,7)`` or ``(7,)``.
This server maps slash keys to LeRobot dotted keys, converts CHW→HWC, resizes to the policy's training resolution
(e.g. 480×640), runs ACT, and returns ``actions`` with shape ``(n_action_steps, 7)`` in **dataset / physical units**
(MEAN_STD un-normalization using dataset stats).
"""

from __future__ import annotations

import argparse
import asyncio
import http
import logging
import time
import traceback
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TVF

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.msgpack_numpy import Packer, unpackb
from lerobot.utils.utils import get_safe_torch_device

logger = logging.getLogger(__name__)


def _maybe_decode_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            out[nk] = _maybe_decode_keys(v)
        return out
    if isinstance(obj, list):
        return [_maybe_decode_keys(x) for x in obj]
    return obj


def _to_hwc_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected image ndim 3, got shape={arr.shape}")
    # HWC / HW RGB
    if arr.shape[-1] in (1, 3, 4):
        return np.ascontiguousarray(arr[..., :3], dtype=np.uint8)
    # CHW (OpenPI / Isaac after resize_with_pad)
    if arr.shape[0] in (1, 3, 4):
        x = np.transpose(arr[:3], (1, 2, 0))
        return np.ascontiguousarray(x, dtype=np.uint8)
    raise ValueError(f"Cannot infer image layout for shape={arr.shape}")


def _resize_hwc(img_hwc: np.ndarray, height: int, width: int) -> np.ndarray:
    t = torch.from_numpy(img_hwc[..., :3]).permute(2, 0, 1).float() / 255.0
    t = TVF.resize(t, (height, width), antialias=True)
    return (t * 255.0).clamp(0, 255).byte().permute(1, 2, 0).contiguous().numpy()


def _squeeze_state(st: np.ndarray) -> np.ndarray:
    st = np.asarray(st, dtype=np.float32).reshape(-1)
    if st.size < 7:
        st = np.pad(st, (0, 7 - st.size))
    return st[:7]


def openpi_obs_to_lerobot_numpy(
    obs: dict[str, Any],
    *,
    exp_h: int,
    exp_w: int,
) -> dict[str, np.ndarray]:
    """Map Isaac/OpenPI-style observation dict to LeRobot numpy observation (HWC uint8 images)."""
    obs = _maybe_decode_keys(obs)

    def pick(keys: list[str]) -> Any:
        for k in keys:
            if k in obs:
                return obs[k]
        if "observation" in obs and isinstance(obs["observation"], dict):
            sub = obs["observation"]
            for k in keys:
                tail = k.split("/")[-1] if "/" in k else k
                tail = tail.replace("observation.", "")
                if tail in sub:
                    return sub[tail]
                if k in sub:
                    return sub[k]
        return None

    st_raw = pick(["observation/state", "observation.state", "state"])
    if st_raw is None:
        raise KeyError("Missing state (expected observation/state or observation.state)")
    st = _squeeze_state(np.asarray(st_raw))

    right_raw = pick(
        [
            "observation/images/right",
            "observation.images.right",
            "observation/wrist_image",
            "wrist_image",
        ]
    )
    if right_raw is None:
        raise KeyError("Missing wrist/right image")
    right_hwc = _to_hwc_rgb_uint8(np.asarray(right_raw))

    top_raw = pick(
        [
            "observation/images/top",
            "observation.images.top",
            "observation/base_camera",
            "base_image",
            "observation/image",
            "image",
        ]
    )
    if top_raw is None:
        top_hwc = right_hwc.copy()
    else:
        top_hwc = _to_hwc_rgb_uint8(np.asarray(top_raw))

    right_hwc = _resize_hwc(right_hwc, exp_h, exp_w)
    top_hwc = _resize_hwc(top_hwc, exp_h, exp_w)

    return {
        "observation.state": st,
        "observation.images.right": right_hwc,
        "observation.images.top": top_hwc,
    }


def unnormalize_mean_std_actions(
    chunk: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> np.ndarray:
    """Invert MEAN_STD: ``(x - mean) / std`` → ``x * std + mean``."""
    # chunk [1, T, D]
    m = mean.to(chunk.device, dtype=chunk.dtype).view(1, 1, -1)
    s = std.to(chunk.device, dtype=chunk.dtype).view(1, 1, -1)
    out = chunk * s + m
    return out.squeeze(0).detach().cpu().numpy().astype(np.float32)


def load_dataset_meta_from_train_config(pretrained_dir: Path) -> tuple[str, str] | None:
    tc = pretrained_dir / "train_config.json"
    if not tc.is_file():
        return None
    import json

    with open(tc, encoding="utf-8") as f:
        cfg = json.load(f)
    ds = cfg.get("dataset") or {}
    rid, root = ds.get("repo_id"), ds.get("root")
    if isinstance(rid, str) and isinstance(root, str):
        return rid, root
    return None


def build_metadata(policy_cfg: PreTrainedConfig) -> dict[str, Any]:
    return {
        "name": "lerobot_act",
        "policy_type": "act",
        "action_dim": policy_cfg.output_features["action"].shape[0],
        "action_horizon": policy_cfg.n_action_steps,
    }


def _summarize_obs_keys(obs: Any) -> str:
    """Human-readable summary of observation dict keys for logging."""
    if not isinstance(obs, dict):
        return f"type={type(obs).__name__!r}"
    parts: list[str] = [f"top_level={list(obs.keys())}"]
    obs_inner = obs.get("observation")
    if isinstance(obs_inner, dict):
        parts.append(f"observation.keys={list(obs_inner.keys())}")
    return " | ".join(parts)


def _extract_task_from_obs(obs: Any) -> str:
    """Extract language/task text from several OpenPI/LeRobot-style locations."""
    if not isinstance(obs, dict):
        return ""

    direct_keys = (
        "prompt",
        "task",
        "language_instruction",
        "instruction",
        "observation/task",
        "observation.prompt",
        "observation.language_instruction",
    )
    for k in direct_keys:
        v = obs.get(k)
        if v is not None:
            return str(v)

    sub = obs.get("observation")
    if isinstance(sub, dict):
        for k in ("prompt", "task", "language_instruction", "instruction"):
            v = sub.get(k)
            if v is not None:
                return str(v)

    return ""


def _round_array(arr: Any, ndigits: int = 3) -> list[float]:
    x = np.asarray(arr, dtype=np.float32).reshape(-1)
    return np.round(x, ndigits).tolist()


def _image_stats(img: np.ndarray) -> str:
    x = np.asarray(img)
    if x.size == 0:
        return f"shape={tuple(x.shape)} empty"
    return (
        f"shape={tuple(x.shape)} "
        f"mean={float(np.mean(x)):.2f} std={float(np.std(x)):.2f} "
        f"min={int(np.min(x))} max={int(np.max(x))}"
    )


def _save_debug_images(debug_dir: Path, request_idx: int, numpy_obs: dict[str, np.ndarray]) -> None:
    """Save the resized images that are actually fed to ACT."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio  # noqa: PLC0415

        for key, name in (
            ("observation.images.right", "right"),
            ("observation.images.top", "top"),
        ):
            img = np.asarray(numpy_obs[key])
            imageio.imwrite(debug_dir / f"obs_{request_idx:06d}_{name}.png", img.astype(np.uint8, copy=False))
    except Exception:
        logger.exception("Failed to save debug images for request %d", request_idx)


async def _health_check(connection, request):  # noqa: ANN001
    try:
        if getattr(request, "path", None) == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
    except Exception:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenPI-compatible WebSocket server for LeRobot ACT.")
    parser.add_argument("--policy.path", dest="policy_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", type=str, default=None)
    parser.add_argument("--dataset.root", dest="dataset_root", type=str, default=None)
    parser.add_argument(
        "--debug_inference",
        action="store_true",
        help=(
            "Log observation state/image statistics and action chunk differences. "
            "Use this to diagnose repeated action chunks / policy collapse."
        ),
    )
    parser.add_argument(
        "--debug_every",
        type=int,
        default=1,
        help="Log debug information every N websocket inference requests.",
    )
    parser.add_argument(
        "--debug_image_dir",
        type=str,
        default=None,
        help="Optional directory to save resized images that are actually fed to ACT.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    debug_every = max(1, int(args.debug_every))
    debug_image_dir = Path(args.debug_image_dir).expanduser().resolve() if args.debug_image_dir else None
    if debug_image_dir is not None:
        debug_image_dir.mkdir(parents=True, exist_ok=True)

    try:
        import websockets.asyncio.server as ws_server  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Install WebSocket extras: pip install 'lerobot[ws-act]' or pip install websockets msgpack"
        ) from e

    pretrained_path = Path(args.policy_path).expanduser().resolve()
    ds_meta_pair = load_dataset_meta_from_train_config(pretrained_path)
    repo_id = args.dataset_repo_id or (ds_meta_pair[0] if ds_meta_pair else None)
    root = args.dataset_root or (ds_meta_pair[1] if ds_meta_pair else None)
    if not repo_id or not root:
        raise ValueError("Provide --dataset.repo_id and --dataset.root or train_config.json beside checkpoint.")

    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    policy_cfg = PreTrainedConfig.from_pretrained(str(pretrained_path))
    policy_cfg.pretrained_path = pretrained_path
    device = get_safe_torch_device(policy_cfg.device, log=True)

    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    policy.eval()

    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(pretrained_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    right_ft = policy_cfg.input_features["observation.images.right"]
    _, exp_h, exp_w = right_ft.shape

    mean = torch.tensor(ds_meta.stats["action"]["mean"], dtype=torch.float32)
    std = torch.tensor(ds_meta.stats["action"]["std"], dtype=torch.float32)

    metadata = build_metadata(policy_cfg)
    packer = Packer()

    infer_lock = asyncio.Lock()

    async def handler(websocket):  # noqa: ANN001
        peer = getattr(websocket, "remote_address", "?")
        logger.info("Connection opened from %s", peer)
        await websocket.send(packer.pack(metadata))

        prev_total_time = None
        prev_state: np.ndarray | None = None
        prev_actions_np: np.ndarray | None = None
        request_idx = 0
        while True:
            try:
                start_time = time.monotonic()
                raw = await websocket.recv()
                if isinstance(raw, str):
                    raise RuntimeError(f"Expected bytes observation, got string: {raw[:200]}")

                obs = unpackb(raw)
                obs = _maybe_decode_keys(obs)
                request_idx += 1

                logger.info("recv obs %s", _summarize_obs_keys(obs))

                infer_t0 = time.monotonic()
                numpy_obs = openpi_obs_to_lerobot_numpy(obs, exp_h=exp_h, exp_w=exp_w)

                task = _extract_task_from_obs(obs)

                if args.debug_inference and (request_idx % debug_every == 0):
                    state_now = numpy_obs["observation.state"].copy()
                    if prev_state is None:
                        state_diff = None
                    else:
                        state_diff = float(np.mean(np.abs(state_now - prev_state)))
                    logger.info(
                        "debug obs req=%d task=%r state=%s state_diff=%s right={%s} top={%s}",
                        request_idx,
                        task,
                        _round_array(state_now),
                        "None" if state_diff is None else f"{state_diff:.6f}",
                        _image_stats(numpy_obs["observation.images.right"]),
                        _image_stats(numpy_obs["observation.images.top"]),
                    )
                    if debug_image_dir is not None:
                        _save_debug_images(debug_image_dir, request_idx, numpy_obs)

                async with infer_lock:
                    transition = copy(numpy_obs)
                    transition = prepare_observation_for_inference(transition, device, task, None)
                    transition = preprocessor(transition)
                    chunk = policy.predict_action_chunk(transition)
                    chunk = chunk[:, : policy_cfg.n_action_steps]
                    actions_np = unnormalize_mean_std_actions(chunk, mean, std)

                if args.debug_inference and (request_idx % debug_every == 0):
                    if prev_actions_np is None:
                        chunk_diff = None
                    elif prev_actions_np.shape == actions_np.shape:
                        chunk_diff = float(np.mean(np.abs(actions_np[:, :6] - prev_actions_np[:, :6])))
                    else:
                        chunk_diff = float("nan")

                    state_for_compare = numpy_obs["observation.state"]
                    compare_dim = min(actions_np.shape[-1], state_for_compare.shape[-1], 7)
                    a0_minus_state = actions_np[0, :compare_dim] - state_for_compare[:compare_dim]
                    mid_idx = actions_np.shape[0] // 2

                    logger.info(
                        "debug act req=%d shape=%s first=%s mid=%s last=%s "
                        "chunk_diff=%s a0_minus_state=%s",
                        request_idx,
                        tuple(actions_np.shape),
                        _round_array(actions_np[0, :7]),
                        _round_array(actions_np[mid_idx, :7]),
                        _round_array(actions_np[-1, :7]),
                        "None" if chunk_diff is None else f"{chunk_diff:.6f}",
                        _round_array(a0_minus_state),
                    )

                prev_state = numpy_obs["observation.state"].copy()
                prev_actions_np = actions_np.copy()

                infer_ms = (time.monotonic() - infer_t0) * 1000

                response = {
                    "actions": actions_np,
                    "server_timing": {
                        "infer_ms": infer_ms,
                    },
                }
                if prev_total_time is not None:
                    response["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                logger.info("send response keys=%s", list(response.keys()))
                await websocket.send(packer.pack(response))
                prev_total_time = time.monotonic() - start_time

            except Exception:
                tb = traceback.format_exc()
                logger.exception("Handler error")
                try:
                    await websocket.send(tb)
                    await websocket.close(code=1011, reason="internal_error")
                except Exception:
                    pass
                break

    async def run_server() -> None:
        async with ws_server.serve(
            handler,
            args.host,
            args.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info("ACT WebSocket server at ws://%s:%s (OpenPI client compatible)", args.host, args.port)
            await server.serve_forever()

    asyncio.run(run_server())


if __name__ == "__main__":
    main()