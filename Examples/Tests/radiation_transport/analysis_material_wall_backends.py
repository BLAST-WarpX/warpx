#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Repeat the originally declared strict CPU/GPU material-wall comparison."""

import argparse
import json
from pathlib import Path

import numpy as np
from analysis_material_wall_control import check, load
from analysis_moving_moment_pulse import plotfiles

parser = argparse.ArgumentParser()
parser.add_argument("reference", type=Path)
parser.add_argument("candidate", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
check(args.reference)
check(args.candidate)
reference = plotfiles(args.reference)
candidate = plotfiles(args.candidate)
assert len(reference) == len(candidate)
history = []
for ref_path, path in zip(reference, candidate, strict=True):
    ref, row = load(ref_path), load(path)
    assert ref["step"] == row["step"]
    assert np.array_equal(ref["ids"], row["ids"])
    assert np.array_equal(ref["mass"], row["mass"])
    scales = {
        "position": 0.001,
        "u": 9e4,
        "rho_nodes": np.max(abs(ref["rho_nodes"])),
        "temperature_nodes": np.max(abs(ref["temperature_nodes"])),
    }
    history.append(
        {
            "step": row["step"],
            "normalized_max_differences": {
                name: float(np.max(abs(row[name] - ref[name])) / scale)
                for name, scale in scales.items()
            },
        }
    )
# Keep the original cross-backend bounds, distinct from the tighter restart gate.
passed = all(
    value < 1e-10
    for row in history
    for value in row["normalized_max_differences"].values()
)
report = {
    "reference": str(args.reference.resolve()),
    "candidate": str(args.candidate.resolve()),
    "all_gates_passed": passed,
    "bound": 1e-10,
    "history": history,
}
(args.output or args.candidate / "backend_comparison.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
assert passed, "Strict cross-backend wall trajectory comparison failed"
