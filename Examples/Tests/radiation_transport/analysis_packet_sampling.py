#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Validate seeded, stratified, conservative multi-packet conversion."""

import argparse
from pathlib import Path

import numpy as np
import yt
from analysis_precision import add_precision_arguments, precision_dtypes
from scipy.constants import c

parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path)
parser.add_argument("--expect-different", action="store_true")
parser.add_argument("--uniform-domain", action="store_true")
parser.add_argument("--domain-offset", type=float, default=0.0)
parser.add_argument("--packets-per-cell", type=int, default=8)
add_precision_arguments(parser)
args = parser.parse_args()
_, particle_dtype, cross_dtype = precision_dtypes(args)


def particles(plotfile: Path) -> np.ndarray:
    data = yt.load(str(plotfile)).all_data()
    columns = [
        data["photons", f"particle_position_{axis}"].to_value("m") for axis in "xyz"
    ]
    columns.extend(
        data["photons", f"particle_momentum_{axis}"].to_value("kg*m/s")
        for axis in "xyz"
    )
    columns.append(data["photons", "particle_weight"].to_ndarray())
    result = np.column_stack(columns)
    return result[np.lexsort(result.T[::-1])]


current = particles(Path("diags/diag1000001"))
converted_cells = 64 if args.uniform_domain else 1
packet_count = args.packets_per_cell
assert packet_count > 0
assert current.shape == (packet_count * converted_cells, 7)
assert np.all(np.isfinite(current))
position_upper = args.domain_offset + (1.0 if args.uniform_domain else 0.5)
assert np.all(
    (current[:, :3] >= args.domain_offset) & (current[:, :3] < position_upper)
)

momentum = current[:, 3:6]
momentum_norm = np.linalg.norm(momentum, axis=1)
represented_energy = np.sum(current[:, 6] * c * momentum_norm)
photon_energy = 1.0e-15
particle_rtol = 3.0e-6 if particle_dtype == np.float32 else 2.0e-12
ledger_rtol = 4.0e-6 if cross_dtype == np.float32 else 3.0e-13
np.testing.assert_allclose(momentum_norm, photon_energy / c, rtol=particle_rtol)

# Exactly one direction lands in every equal-area cos(theta) stratum.
mu = momentum[:, 2] / momentum_norm
strata = np.floor(packet_count * (mu + 1.0) / 2.0).astype(int)
np.testing.assert_array_equal(
    np.sort(strata), np.repeat(np.arange(packet_count), converted_cells)
)
if args.uniform_domain:
    cell_indices = np.floor(4.0 * (current[:, :3] - args.domain_offset)).astype(int)
    assert np.all((cell_indices >= 0) & (cell_indices < 4))
    cell_ids = np.ravel_multi_index(cell_indices.T, (4, 4, 4))
    np.testing.assert_array_equal(np.bincount(cell_ids, minlength=64), packet_count)
    for cell_id in range(64):
        np.testing.assert_array_equal(
            np.sort(strata[cell_ids == cell_id]), np.arange(packet_count)
        )

# The one-cell case converts 1e-13 J. The uniform 4^3 reference fills the
# unit-volume domain at 8e-13 J/m^3, independent of its decomposition.
radiation = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
assert radiation.shape == (2, 17)
initial_energy = 8.0e-13 if args.uniform_domain else 1.0e-13
np.testing.assert_allclose(represented_energy, initial_energy, rtol=ledger_rtol)
np.testing.assert_allclose(radiation[:, 2], initial_energy, rtol=ledger_rtol)
np.testing.assert_allclose(radiation[-1, 3], initial_energy, rtol=ledger_rtol)
np.testing.assert_allclose(
    radiation[-1, 4], 0.0, atol=64.0 * np.finfo(cross_dtype).eps * initial_energy
)
np.testing.assert_allclose(radiation[:, 5:17], 0.0, atol=1.0e-27)

if args.reference is not None:
    reference = particles(args.reference / "diags/diag1000001")
    assert reference.shape == current.shape
    if args.expect_different:
        reference_energy = np.sum(
            reference[:, 6] * c * np.linalg.norm(reference[:, 3:6], axis=1)
        )
        np.testing.assert_allclose(
            represented_energy, reference_energy, rtol=ledger_rtol
        )
        assert not np.array_equal(current[:, :6], reference[:, :6])
    else:
        np.testing.assert_array_equal(current, reference)

print(
    f"multi-packet conversion: count={len(current)}, energy={represented_energy:.16e} J"
)
