#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Reject changed or malformed private Qei/FV restart contracts."""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("executable", type=Path)
parser.add_argument("inputs", type=Path)
parser.add_argument("checkpoint", type=Path)
args = parser.parse_args()
executable, inputs, checkpoint = (
    path.resolve() for path in (args.executable, args.inputs, args.checkpoint)
)
support = "HybridQeiSupport.txt"
transport = "HybridIdealElectronTransport.txt"
cases = (
    (
        "missing_support",
        support,
        None,
        (),
        "Restart must preserve the Qei thermal support",
    ),
    (
        "old_support",
        support,
        "resolved_pairwise_v2\n",
        (),
        "Invalid resolved Qei support",
    ),
    (
        "negative_counter",
        support,
        "resolved_pairwise_v3 1 -1\n",
        (),
        "Invalid resolved Qei support",
    ),
    (
        "overflow_counter",
        support,
        "resolved_pairwise_v3 1 18446744073709551616\n",
        (),
        "Invalid resolved Qei support",
    ),
    (
        "trailing_support",
        support,
        "resolved_pairwise_v3 1 10 extra\n",
        (),
        "Invalid resolved Qei support",
    ),
    (
        "missing_transport",
        transport,
        None,
        (),
        "Restart must preserve the ideal electron transport",
    ),
    (
        "bad_transport",
        transport,
        "ideal_finite_volume_v1 extra\n",
        (),
        "Invalid ideal electron transport",
    ),
    (
        "changed_seed",
        None,
        None,
        ("hybrid_pic_model.resolved_qei_seed=2",),
        "Invalid resolved Qei support",
    ),
    (
        "changed_support",
        None,
        None,
        ("hybrid_pic_model.resolved_qei_support=0",),
        "Restart must preserve the Qei thermal support",
    ),
)
for name, filename, replacement, overrides, message in cases:
    # Mutate only a fresh, task-owned copy; never the producer checkpoint.
    with tempfile.TemporaryDirectory(prefix=f"{name}_", dir=Path.cwd()) as directory:
        root = Path(directory)
        copied = root / "checkpoint"
        shutil.copytree(checkpoint, copied)
        if filename:
            target = copied / filename
            if replacement is None:
                target.unlink()
            else:
                target.write_text(replacement)
        result = subprocess.run(
            [
                str(executable),
                str(inputs),
                f"amr.restart={copied}",
                "diagnostics.enable=0",
                "amrex.the_arena_init_size=0",
                *overrides,
            ],
            cwd=root,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=45,
        )
        assert result.returncode != 0 and message in result.stdout, (
            name,
            result.stdout,
        )
        print(f"{name}: rejected with the intended contract diagnostic")
