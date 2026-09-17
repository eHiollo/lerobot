#!/usr/bin/env python

"""Non-blocking JSONL diagnostics for XLeVR position-control tuning."""

from __future__ import annotations

import copy
import json
import logging
import math
import queue
import threading
from pathlib import Path
from typing import Any


_STOP = object()


def _json_safe(value: Any) -> Any:
    """Convert common numeric/container values to strict JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    return numeric if math.isfinite(numeric) else None


def get_xlevr_diagnostics(pipeline: Any) -> dict[str, Any] | None:
    """Return the newest mapper snapshot without coupling callers to step position."""
    for step in getattr(pipeline, "steps", ()):
        getter = getattr(step, "get_last_diagnostics", None)
        if getter is not None:
            return getter()
    return None


class XLeVRDiagnosticsWriter:
    """Write diagnostics on a worker thread; a full queue drops instead of blocking control."""

    def __init__(self, path: str | Path, *, queue_size: int = 2048, flush_every: int = 30):
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1")
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8")
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._flush_every = flush_every
        self._closed = False
        self._error: Exception | None = None
        self.records_written = 0
        self.dropped_records = 0
        self._record_sequence = 0
        self._thread = threading.Thread(
            target=self._run,
            name="xlevr-diagnostics-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def error(self) -> Exception | None:
        return self._error

    def write(self, record: dict[str, Any]) -> bool:
        if self._closed or self._error is not None:
            self.dropped_records += 1
            return False
        queued = copy.deepcopy(record)
        queued["diagnostics_record_sequence"] = self._record_sequence
        self._record_sequence += 1
        try:
            self._queue.put_nowait(queued)
        except queue.Full:
            self.dropped_records += 1
            return False
        return True

    def _run(self) -> None:
        pending_since_flush = 0
        try:
            while True:
                record = self._queue.get()
                try:
                    if record is _STOP:
                        break
                    line = json.dumps(
                        _json_safe(record),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    self._stream.write(line + "\n")
                    self.records_written += 1
                    pending_since_flush += 1
                    if pending_since_flush >= self._flush_every:
                        self._stream.flush()
                        pending_since_flush = 0
                finally:
                    self._queue.task_done()
        except Exception as exc:  # Diagnostics must never stop the control loop.
            self._error = exc
            logging.exception("XLeVR diagnostics writer failed: %s", exc)
        finally:
            self._stream.flush()
            self._stream.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread.is_alive():
            self._queue.put(_STOP)
            self._thread.join()

    def __enter__(self) -> "XLeVRDiagnosticsWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
