#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL

import sys

import numpy as np
import yt
from analysis_nonlinear_electron_conduction import reference


def fields(path):
    dataset = yt.load(path)
    grid = dataset.covering_grid(0, dataset.domain_left_edge, dataset.domain_dimensions)
    return np.asarray(grid["boxlib", "Te"]).reshape(-1), np.asarray(
        grid["boxlib", "rho"]
    ).reshape(-1)


initial, rho_initial = fields(sys.argv[1])
final, rho_final = fields(sys.argv[2])
assert initial.shape == final.shape == (64,)
assert np.all(np.isfinite(final)) and np.all(final > 0)
charge = 20 * 1.602176634e-19 * 1000 / (183.84 * 1.66053906892e-27)
np.testing.assert_allclose(rho_initial, charge, rtol=1.0e-12)
np.testing.assert_allclose(rho_final, rho_initial, rtol=1.0e-12)

# Plotfile Te is the arithmetic nodal-to-cell average, not an EOS inversion
# of averaged internal energy. Apply that same sampling to the PDE reference.
nodes = np.arange(64) / 64
initial_nodes = 1.0e4 * (1 + 0.5 * np.cos(2 * np.pi * nodes))
expected_initial = (initial_nodes + np.roll(initial_nodes, -1)) / 2
np.testing.assert_allclose(initial, expected_initial, rtol=1.0e-12)
reference_nodes = reference(512, True)[::8]
expected = (reference_nodes + np.roll(reference_nodes, -1)) / 2
error = np.linalg.norm(final - expected) / np.linalg.norm(expected - initial)
assert np.linalg.norm(final - initial) > 0.1 * np.linalg.norm(initial - initial.mean())
assert error < 0.03, error
print(
    f"Native fixed-Z=20 table conduction: profile error={error:.6g}; physical density unchanged"
)
