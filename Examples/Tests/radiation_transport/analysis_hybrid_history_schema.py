#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Exercise new/legacy/invalid native moment-history checkpoints on private copies."""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from analysis_material_wall_control import load
from analysis_moving_moment_interface import nodes


def assert_exact(actual, expected):
    # These tests are registered only for double fields/particles. Compare
    # native stored bits as well as values, including the sign of zero.
    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float64).view(np.uint64),
        np.asarray(expected, dtype=np.float64).view(np.uint64),
    )


parser = argparse.ArgumentParser()
parser.add_argument("executable", type=Path)
parser.add_argument("input", type=Path)
parser.add_argument("source", type=Path)
parser.add_argument("--probe-executable", type=Path)
args = parser.parse_args()
root = Path(tempfile.mkdtemp(prefix="hybrid-history-schema-", dir=Path.cwd()))
for case, step, message in (
    ("valid", 1600, None),
    ("initial", 0, None),
    ("legacy", 1600, None),
    ("changed_shape", 1600, "deposition contract changed"),
    ("missing_manifest", 1600, "has data but no manifest"),
    ("missing_field", 1600, "Incomplete or inconsistent"),
    ("trailing", 1600, "Trailing hybrid moment history"),
):
    work = root / case
    work.mkdir()
    checkpoint = work / "checkpoint"
    shutil.copytree(args.source / "diags" / f"chk{step:06d}", checkpoint)
    manifest = checkpoint / "HybridMomentHistory.txt"
    assert manifest.is_file(), "Source must be a new-schema checkpoint"
    assert manifest.read_text().split()[1] == ("0" if step == 0 else "1")
    if case == "legacy":
        for path in checkpoint.glob("HybridMomentHistory*"):
            path.rename(path.with_name(path.name + ".held"))
    elif case == "missing_manifest":
        manifest.rename(manifest.with_suffix(".held"))
    elif case == "missing_field":
        field = checkpoint / "HybridMomentHistory_rho_H"
        field.rename(field.with_suffix(".held"))
    elif case == "trailing":
        manifest.write_text(manifest.read_text() + "unexpected\n")
    result = subprocess.run(
        [
            str(args.executable.resolve()),
            str(args.input.resolve()),
            "hybrid_pic_model.conservative_pressure_work_pec=1",
            f"amr.restart={checkpoint}",
            f"max_step={step + 1}",
            "diagnostics.enable=0",
        ]
        + (["algo.particle_shape=3"] if case == "changed_shape" else []),
        cwd=work,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    (work / "run.log").write_text(result.stdout)
    if message is None:
        assert result.returncode == 0, f"{case}: see {work / 'run.log'}"
    else:
        assert result.returncode != 0 and message in result.stdout, (
            f"{case}: missing intended rejection; see {work / 'run.log'}"
        )
    print(f"{case}: passed")
print(f"Checkpoint copies and logs retained in {root}")

if args.probe_executable:
    work = root / "exact_history_probe"
    work.mkdir()
    checkpoint = args.source.resolve() / "diags/chk001600"
    result = subprocess.run(
        [
            str(args.probe_executable.resolve()),
            str(args.input.resolve()),
            "hybrid_pic_model.conservative_pressure_work_pec=1",
            "test.hybrid_restore_probe=1",
            f"amr.restart={checkpoint}",
        ],
        cwd=work,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    (work / "run.log").write_text(result.stdout)
    assert result.returncode == 0, f"Probe failed: see {work / 'run.log'}"
    plot = args.source / "diags/diag1001600"
    baseline = load(plot)
    cells = len(baseline["rho_nodes"]) - 1
    for name, expected in (
        ("rho", baseline["rho_nodes"]),
        ("temperature", baseline["temperature_nodes"]),
    ):
        assert_exact(
            nodes(work / f"probe_{name}_H", cells, periodic=False),
            expected,
        )
    for d, axis in enumerate("xyz"):
        assert_exact(
            nodes(work / f"probe_current_{d}_H", cells, periodic=False),
            nodes(plot / f"raw_fields/Level_0/j{axis}_fp_H", cells, periodic=False),
        )
    print("Restored charge, temperature and current history are bitwise exact")
