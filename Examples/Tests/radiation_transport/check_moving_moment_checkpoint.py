#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Reject incomplete moving-model checkpoints without modifying the producer."""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def check(executable, checkpoint, missing):
    checkpoint = checkpoint.resolve(strict=True)
    relative = (
        Path("RadiationMomentModel_data.txt")
        if missing == "model"
        else Path("Level_0") / f"radiation_moment_q{missing}[level=0]_H"
    )
    assert (checkpoint / relative).is_file()
    # Only the disposable copy is corrupted; retain the failed child's output.
    with tempfile.TemporaryDirectory(prefix="moving-checkpoint-", dir=".") as temporary:
        candidate = Path(temporary).resolve() / "checkpoint"
        shutil.copytree(checkpoint, candidate)
        (candidate / relative).unlink()
        result = subprocess.run(
            [
                str(executable.resolve(strict=True)),
                "inputs_base_1d_moving_moment_pulse",
                "max_step=201",
                "amr.n_cell=128",
                "amr.max_grid_size=64",
                "warpx.const_dt=8.333333333333333e-12",
                f"amr.restart={candidate}",
                "amrex.throw_exception=1",
                "amrex.the_arena_init_size=0",
                "warpx.verbose=1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=60,
        )
    Path(f"missing_{missing}.log").write_text(result.stdout)
    expected = (
        "Restart must preserve the radiation moment model"
        if missing == "model"
        else "Checkpoint is missing the required MultiFab header"
    )
    assert result.returncode != 0, result.stdout
    assert expected in result.stdout, result.stdout
    if missing != "model":
        assert relative.name in result.stdout, result.stdout
    assert "STEP 201 ends" not in result.stdout, result.stdout
    print(f"Missing {missing}: rejected before advancing the restarted state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("missing", choices=("model", "x", "y", "z"))
    args = parser.parse_args()
    check(args.executable, args.checkpoint, args.missing)
