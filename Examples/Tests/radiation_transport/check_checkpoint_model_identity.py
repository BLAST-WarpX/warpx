#!/usr/bin/env python3
"""Continue populated radiation fields and reject model changes at restart."""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(40)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("groups", "opacity"))
parser.add_argument("executable", type=Path)
parser.add_argument("inputs", type=Path)
parser.add_argument("launcher", nargs=argparse.REMAINDER)
args = parser.parse_args()
executable = str(args.executable.resolve())


def run(directory, inputs, overrides, *, launcher=(), refusal=None):
    directory.mkdir()
    result = subprocess.run(
        [*launcher, executable, str(inputs), *overrides],
        cwd=directory,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    output = " ".join(re.sub(r"\n\s*#\s+", " ", result.stdout).lower().split())
    if refusal is None:
        assert result.returncode == 0, result.stdout[-5000:]
    else:
        assert result.returncode != 0 and refusal in output, (
            refusal,
            result.stdout[-5000:],
        )
    return directory


def fields(directory, step, names):
    ds = yt.load(str(directory / f"diags/diag1{step:06d}"))
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    return [np.asarray(grid["boxlib", name].v) for name in names]


with tempfile.TemporaryDirectory(
    prefix="radiation_model_identity_", dir="."
) as temporary:
    root = Path(temporary).resolve()
    inputs = root / "inputs"
    deck = args.inputs.read_text()
    if args.mode == "groups":
        step = 1
        deck = deck.replace(
            "diagnostics.diags_names = diag1", "diagnostics.diags_names = diag1 chk"
        )
        deck += """
warpx.random_seed = 42
chk.diag_type = Full
chk.format = checkpoint
chk.intervals = 1
chk.file_prefix = diags/chk
"""
        observed = ["radiation_diffusion_energy_g0", "radiation_diffusion_energy_g1"]
        incompatible = {
            "edge": "radiation_transport.energy_group_boundaries=1.5*group_boundary",
            "energy": "radiation_transport.group_photon_energies=0.6*group_boundary 2*group_boundary",
        }
        message = "restart changed its group edges or representative photon energies"
    else:
        step = 4
        old_expression = "chi*(1+0.05*sin(2*pi*t/(8*dt0)))"
        assert deck.count(old_expression) == 1
        deck = deck.replace(old_expression, "if(t < 4.5*dt0, chi, chi*after_gain)")
        deck += "\nmy_constants.after_gain = 1\n"
        observed = ["radiation_resolved_opacity", "radiation_diffusion_energy"]
        incompatible = {"future": "my_constants.after_gain=1.03125"}
        message = "restart changed mask or opacity parser constants"
    inputs.write_text(deck)
    limits = [f"max_step={step + 1}", "amrex.call_addr2line=0"]
    original = run(root / "continuous", inputs, limits)
    checkpoint = original / f"diags/chk{step:06d}"
    assert checkpoint.is_dir()
    resume = [*limits, f"amr.restart={checkpoint}"]
    restarted = run(root / "restarted", inputs, resume, launcher=args.launcher)
    reference = fields(original, step + 1, observed)
    actual = fields(restarted, step + 1, observed)
    checkpoint_fields = fields(original, step, observed)
    assert all(np.isfinite(value).all() for value in checkpoint_fields)
    assert np.any(checkpoint_fields[0] > 0)
    for expected, value in zip(reference, actual):
        if args.launcher:
            np.testing.assert_allclose(value, expected, rtol=2e-12, atol=0)
        else:
            np.testing.assert_array_equal(value, expected)
    for case, override in incompatible.items():
        # The legacy switch must never override an identity that is present.
        run(
            root / case,
            inputs,
            [*resume, override, "radiation_transport.assume_legacy_checkpoint_model=1"],
            launcher=args.launcher,
            refusal=message,
        )
    if args.mode == "groups" and not args.launcher:
        legacy = root / "legacy"
        shutil.copytree(checkpoint, legacy)
        manifest = legacy / "RadiationEnergyGroups_data.txt"
        manifest.unlink()
        resume_legacy = [*limits, f"amr.restart={legacy}"]
        run(
            root / "unverified",
            inputs,
            resume_legacy,
            refusal="missing spectral group identity",
        )
        migrated = run(
            root / "verified",
            inputs,
            [*resume_legacy, "radiation_transport.assume_legacy_checkpoint_model=1"],
        )
        assert (migrated / f"diags/chk{step + 1:06d}" / manifest.name).read_bytes() == (
            checkpoint / manifest.name
        ).read_bytes()
        manifest.write_text((checkpoint / manifest.name).read_text() + "unexpected\n")
        run(root / "damaged", inputs, resume_legacy, refusal="unexpected trailing data")
