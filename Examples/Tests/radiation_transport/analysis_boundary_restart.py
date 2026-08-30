#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Checkpoint/restart equivalence for nonzero radiation-boundary ledgers."""

import argparse
from pathlib import Path

import numpy as np
from scipy.constants import c

parser = argparse.ArgumentParser()
parser.add_argument("--compare-reference", action="store_true")
parser.add_argument("--precision", choices=["SINGLE", "DOUBLE"], default="DOUBLE")
args = parser.parse_args()

physics_rtol = 2.0e-6 if args.precision == "SINGLE" else 2.0e-12


def load_table(path):
    return np.atleast_2d(np.loadtxt(path))


radiation = load_table("diags/radiation_energy.txt")
momentum = load_table("diags/radiation_momentum.txt")
assert radiation.shape[1] == 17
assert momentum.shape[1] == 20
assert radiation[-1, 14] > 0.0
np.testing.assert_allclose(radiation[:, 11:13], 0.0, atol=1.0e-30)
np.testing.assert_allclose(radiation[:, 13], radiation[:, 7], rtol=physics_rtol)
np.testing.assert_allclose(radiation[:, 14], radiation[:, 8], rtol=physics_rtol)
np.testing.assert_allclose(radiation[:, 15:17], 0.0, atol=1.0e-30)
np.testing.assert_allclose(momentum[:, 14:20], 0.0, atol=1.0e-24)
np.testing.assert_allclose(
    momentum[:, 13], radiation[:, 14] / c, rtol=physics_rtol, atol=1.0e-24
)

if args.compare_reference:
    reference_dir = Path.cwd().with_name(Path.cwd().name.removesuffix("_restart"))
    reference_radiation = load_table(reference_dir / "diags/radiation_energy.txt")
    reference_momentum = load_table(reference_dir / "diags/radiation_momentum.txt")
    reference_energy_by_step = {int(row[0]): row for row in reference_radiation}
    reference_momentum_by_step = {int(row[0]): row for row in reference_momentum}
    for row in radiation:
        np.testing.assert_allclose(
            row,
            reference_energy_by_step[int(row[0])],
            rtol=2.0e-12,
            atol=1.0e-20,
        )
    for row in momentum:
        np.testing.assert_allclose(
            row,
            reference_momentum_by_step[int(row[0])],
            rtol=2.0e-12,
            atol=1.0e-20,
        )
else:
    transport_checkpoint = Path("diags/chk000006/RadiationTransport_data.txt")
    assert transport_checkpoint.is_file()
    checkpoint_values = transport_checkpoint.read_text().split()
    assert len(checkpoint_values) in (1, 2)
    checkpoint_cumulative_escape = float(checkpoint_values[0])
    checkpoint_cumulative_residual = (
        float(checkpoint_values[1]) if len(checkpoint_values) == 2 else 0.0
    )
    assert np.isfinite(checkpoint_cumulative_residual)
    step_six = radiation[np.asarray(radiation[:, 0], dtype=int) == 6]
    assert step_six.shape[0] == 1
    np.testing.assert_allclose(
        checkpoint_cumulative_escape, step_six[0, 8], rtol=2.0e-12
    )
    np.testing.assert_allclose(
        checkpoint_cumulative_residual, step_six[0, 16], rtol=2.0e-12, atol=1.0e-30
    )
