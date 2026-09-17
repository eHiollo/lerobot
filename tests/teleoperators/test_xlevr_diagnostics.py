#!/usr/bin/env python

import json

import numpy as np

from lerobot.teleoperators.xlevr.diagnostics import (
    XLeVRDiagnosticsWriter,
    get_xlevr_diagnostics,
)


def test_jsonl_writer_drains_and_converts_nonfinite_values(tmp_path):
    output_path = tmp_path / "meta" / "xlevr_diagnostics" / "session.jsonl"
    writer = XLeVRDiagnosticsWriter(output_path, queue_size=8, flush_every=2)

    assert writer.write({"sample": 1, "vector": np.array([1.0, np.nan])})
    assert writer.write({"sample": 2, "value": float("inf")})
    writer.close()

    records = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "sample": 1,
            "vector": [1.0, None],
            "diagnostics_record_sequence": 0,
        },
        {
            "sample": 2,
            "value": None,
            "diagnostics_record_sequence": 1,
        },
    ]
    assert writer.records_written == 2
    assert writer.dropped_records == 0
    assert writer.error is None
    assert writer.write({"late": True}) is False
    assert writer.dropped_records == 1


def test_get_xlevr_diagnostics_finds_mapper_step_and_returns_copy():
    snapshot = {"guard_reason": "active"}

    class Step:
        def get_last_diagnostics(self):
            return dict(snapshot)

    class Pipeline:
        steps = [object(), Step()]

    found = get_xlevr_diagnostics(Pipeline())
    found["guard_reason"] = "changed"
    assert snapshot["guard_reason"] == "active"
