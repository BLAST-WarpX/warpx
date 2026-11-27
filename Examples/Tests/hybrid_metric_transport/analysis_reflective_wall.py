#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Independent CIC particle-to-node charge oracle at RZ reflecting walls."""

import sys
from pathlib import Path

import numpy as np
import yt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "radiation_transport"))
from analysis_moving_moment_interface import nodes

yt.set_log_level(50)
profiles = []
for step in (0, 4):
    path = Path(f"diags/plt{step:06d}")
    ds = yt.load(str(path))
    nr, nz = np.asarray(ds.domain_dimensions[:2], dtype=int)
    dr, dz = np.asarray(ds.domain_width[:2], dtype=float) / [nr, nz]
    particles = ds.all_data()
    radius = np.asarray(particles["ions", "particle_position_x"])
    axial = np.asarray(particles["ions", "particle_position_y"])
    charge = np.asarray(particles["ions", "particle_weight"]) * 1.602176634e-19
    ri, zi = radius / dr, axial / dz
    ir, iz = np.floor(ri).astype(int), np.floor(zi).astype(int)
    fr, fz = ri - ir, zi - iz
    oracle = np.zeros((nr + 1, nz + 1))
    for di in (0, 1):
        for dj in (0, 1):
            contribution = charge * (fr if di else 1 - fr) * (fz if dj else 1 - fz)
            np.add.at(oracle, (ir + di, iz + dj), contribution)
    native_volume = np.broadcast_to(
        (2 * np.pi * np.arange(nr + 1) * dr**2 * dz)[:, None], oracle.shape
    ).copy()
    native_volume[0, :] = np.pi * dr**2 * dz / 3
    native_volume[-1, :] *= 0.5
    native_volume[:, [0, -1]] *= 0.5
    oracle /= native_volume
    actual = nodes(path / "raw_fields/Level_0/rho_fp_H", [nr, nz], periodic=False)
    error = np.max(abs(actual - oracle)) / np.max(abs(oracle))
    assert error < 1.0e-11, error
    charge_error = abs(np.sum(actual * native_volume) / np.sum(charge) - 1)
    assert charge_error < 1.0e-11, charge_error
    print(
        f"Step {step}: deposited charge error {error}, integrated error {charge_error}"
    )
    profiles.append(actual)
assert np.max(abs(profiles[1] - profiles[0])) / np.max(abs(profiles[0])) > 1.0e-4
