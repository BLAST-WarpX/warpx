#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Reject incomplete moving-model checkpoints without modifying the producer."""

import argparse
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path


def run_rejection(command, log, timeout=60):
    """Wait for the actual child, not pipe EOF inherited by MPI helpers.

    Keep output even on timeout and clean up only the process group created for
    this invocation. A timeout remains a failure, never an accepted rejection.
    """
    timed_out = False
    with log.open("w") as output:
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.kill()
            process.wait(timeout=10)
    text = log.read_text()
    assert not timed_out, f"Checkpoint rejection exceeded {timeout} seconds:\n{text}"
    return subprocess.CompletedProcess(command, process.returncode, text)


def check(executable, checkpoint, missing, rz=False):
    checkpoint = checkpoint.resolve(strict=True)
    manifests = {
        "model": Path("RadiationMomentModel_data.txt"),
        "ledger": Path("RadiationMomentTransportLedger_data.txt"),
        "cartesian_model": Path("RadiationMomentModel_data.txt"),
    }
    relative = manifests.get(
        missing, Path("Level_0") / f"radiation_moment_q{missing}[level=0]_H"
    )
    assert (checkpoint / relative).is_file()
    # Only the disposable copy is corrupted; retain the failed child's output.
    with tempfile.TemporaryDirectory(prefix="moving-checkpoint-", dir=".") as temporary:
        candidate = Path(temporary).resolve() / "checkpoint"
        shutil.copytree(checkpoint, candidate)
        if missing == "cartesian_model":
            assert rz
            (candidate / relative).write_text(
                "gray_m1_low_beta_nodal_shape_ledger_v3 1\n"
            )
        else:
            (candidate / relative).unlink()
        next_step = 129 if rz else 201
        configuration = (
            [
                "inputs_base_rz_moving_moment_pulse",
                "test.radiation_runtime=1",
                "test.moment_runtime=1",
            ]
            if rz
            else [
                "inputs_base_1d_moving_moment_pulse",
                "amr.n_cell=128",
                "amr.max_grid_size=64",
                "warpx.const_dt=8.333333333333333e-12",
            ]
        )
        result = run_rejection(
            [
                str(executable.resolve(strict=True)),
                *configuration,
                f"max_step={next_step}",
                f"amr.restart={candidate}",
                # These standalone C++ drivers have no exception handler.
                # Use AMReX's MPI-aware abort path, not std::terminate during
                # an uncaught assertion exception with OpenMP/MPI active.
                "amrex.throw_exception=0",
                "amrex.the_arena_init_size=0",
                "warpx.verbose=1",
            ],
            Path(f"missing_{missing}.log"),
        )
    expected = {
        "model": "Restart must preserve the radiation moment model",
        "ledger": "Moving radiation checkpoint must preserve its declared transport",
        "cartesian_model": "RZ moment restart requires the meridional nearest-cell model",
    }.get(missing, "Checkpoint is missing the required MultiFab header")
    assert result.returncode != 0, result.stdout
    assert expected in result.stdout, result.stdout
    if missing not in manifests:
        assert relative.name in result.stdout, result.stdout
    assert f"STEP {next_step} ends" not in result.stdout, result.stdout
    print(f"Missing {missing}: rejected before advancing the restarted state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "missing", choices=("model", "ledger", "x", "y", "z", "cartesian_model")
    )
    parser.add_argument("--rz", action="store_true")
    args = parser.parse_args()
    check(args.executable, args.checkpoint, args.missing, args.rz)
