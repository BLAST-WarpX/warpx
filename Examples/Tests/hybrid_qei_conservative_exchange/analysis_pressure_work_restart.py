#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Compare the same native RZ pressure-work trajectory after checkpoint restart."""

import argparse
import json
from pathlib import Path

import numpy as np
import yt

parser = argparse.ArgumentParser()
parser.add_argument("reference", type=Path)
parser.add_argument("candidate", type=Path)
args = parser.parse_args()
yt.set_log_level(50)


def load(directory):
    dataset = yt.load(str(directory / "diags/plt000400"))
    grid = dataset.covering_grid(0, dataset.domain_left_edge, dataset.domain_dimensions)
    fields = {name: grid["boxlib", name].v for name in ("rho", "Te", "Pe")}
    data = dataset.all_data()
    order = np.lexsort((data["ions", "particle_cpu"].v, data["ions", "particle_id"].v))
    for species, name in dataset.field_list:
        if species == "ions":
            fields[name] = data[species, name].v[order]
    return fields


reference, candidate = load(args.reference), load(args.candidate)
assert reference.keys() == candidate.keys()
report = {}
passed = True
for name, expected in reference.items():
    actual = candidate[name]
    scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
    difference = float(np.max(np.abs(actual - expected)))
    if name in ("particle_id", "particle_cpu", "particle_weight"):
        valid = np.array_equal(actual, expected)
    else:
        valid = np.allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12 * scale)
    report[name] = {
        "relative_max_difference": difference / scale,
        "passed": bool(valid),
    }
    passed &= valid
Path("pressure_work_restart_comparison.json").write_text(
    json.dumps({"all_gates_passed": bool(passed), "quantities": report}, indent=2)
    + "\n"
)
assert passed, report
print("RZ pressure-work restart preserves the native fields and particle trajectory.")
