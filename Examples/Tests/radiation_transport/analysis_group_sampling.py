#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Check conservative sampling of a weak band without hiding it in total energy."""

import argparse
from pathlib import Path

import numpy as np
import yt
from analysis_precision import add_precision_arguments, precision_dtypes
from scipy.constants import c

parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path)
parser.add_argument("--cap", type=int, default=4096)
parser.add_argument("--streaming", action="store_true")
add_precision_arguments(parser)
args = parser.parse_args()
_, particle_dtype, cross_dtype = precision_dtypes(args)
rtol = 4.0e-6 if cross_dtype == np.float32 else 3.0e-13


def packets(directory):
    data = yt.load(str(directory / "diags/diag1000001")).all_data()
    columns = [data["photons", f"particle_position_{a}"].to_value("m") for a in "xyz"]
    columns += [
        data["photons", f"particle_momentum_{a}"].to_value("kg*m/s") for a in "xyz"
    ]
    columns.append(data["photons", "particle_weight"].to_ndarray())
    result = np.column_stack(columns)
    return result[np.lexsort(result.T[::-1])]


current = packets(Path("."))
assert np.all(np.isfinite(current))
if args.streaming:
    # The injected packet points exactly along +z; seeded conversion directions
    # do not. Identify it geometrically rather than by a spectral energy cutoff.
    existing = (current[:, 3] == 0) & (current[:, 4] == 0)
    assert np.count_nonzero(existing) == 1
    np.testing.assert_allclose(
        c * np.linalg.norm(current[existing, 3:6], axis=1) * current[existing, 6],
        8.0e-19,
        rtol=rtol,
    )
    current = current[~existing]
energy = c * np.linalg.norm(current[:, 3:6], axis=1)
groups = (energy >= 2.0e-15).astype(int)
assert np.all((current[:, :3] >= 0) & (current[:, :3] < 1))
cell_ids = np.ravel_multi_index(np.floor(4 * current[:, :3]).astype(int).T, (4, 4, 4))
for group, (count, expected_energy) in enumerate(((4, 8.0e-13), (8, 8.0e-19))):
    if args.streaming and group == 1:
        count //= 2
    count = min(count, args.cap)
    selected = groups == group
    np.testing.assert_array_equal(np.bincount(cell_ids[selected], minlength=64), count)
    np.testing.assert_allclose(energy[selected], (1.0e-15, 4.0e-15)[group], rtol=rtol)
    np.testing.assert_allclose(
        np.sum(energy[selected] * current[selected, 6]), expected_energy, rtol=rtol
    )
    mu = current[selected, 5] * c / energy[selected]
    strata = np.floor(count * (mu + 1) / 2).astype(int)
    for cell_id in range(64):
        np.testing.assert_array_equal(
            np.sort(strata[cell_ids[selected] == cell_id]), np.arange(count)
        )

ledger = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
np.testing.assert_allclose(
    ledger[:, 2], 8.0e-13 + (2 if args.streaming else 1) * 8.0e-19, rtol=rtol
)
np.testing.assert_allclose(ledger[-1, 3], ledger[-1, 2], rtol=rtol)
np.testing.assert_allclose(ledger[-1, 4], 0, atol=1.0e-27)
if args.reference is not None:
    np.testing.assert_array_equal(current, packets(args.reference))
print("Both spectral bands retain their allocated counts and represented energy.")
