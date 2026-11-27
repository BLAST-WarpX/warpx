#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""A resolved cubic tail survives a native moving-particle vacuum crossing."""

import sys
from pathlib import Path

import numpy as np
import yt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "radiation_transport"))
from analysis_moving_moment_interface import nodes

yt.set_log_level(50)
nr, nz = 8, 16
dr, dz = 0.001 / nr, 0.001 / nz
volume = 2 * np.pi * np.arange(nr + 1) * dr**2 * dz
volume[0] = np.pi * dr**2 * dz / 3
volume[-1] *= 0.5


def shape(position, spacing, count):
    coordinate = position / spacing
    cell = int(np.floor(coordinate))
    fraction = coordinate - cell
    result = np.zeros(count + 1)
    result[cell - 1 : cell + 3] = (
        (1 - fraction) ** 3 / 6,
        2 / 3 - fraction**2 * (1 - fraction / 2),
        2 / 3 - (1 - fraction) ** 2 * (1 - (1 - fraction) / 2),
        fraction**3 / 6,
    )
    return result


for step in (0, 10):
    plot = Path(f"diags/plt{step:06d}")
    ds = yt.load(str(plot))
    particle = ds.all_data()
    assert len(particle["ions", "particle_weight"]) == 1
    radius = float(particle["ions", "particle_position_x"][0])
    axial = float(particle["ions", "particle_position_y"][0])
    charge = float(particle["ions", "particle_weight"][0]) * 1.602176634e-19
    rho = nodes(plot / "raw_fields/Level_0/rho_fp_H", [nr, nz], [False, True])
    expected = charge * shape(radius, dr, nr)[:, None] * shape(axial, dz, nz)[None, :-1]
    np.testing.assert_allclose(rho * volume[:, None], expected, rtol=1.0e-7, atol=0)
    np.testing.assert_allclose(np.sum(rho * volume[:, None]), charge, rtol=1.0e-12)
    temperature = nodes(
        Path(f"diags/chk{step:06d}/Level_0/hybrid_electron_temperature_fp[level=0]_H"),
        [nr, nz],
        [False, True],
    )
    assert np.all(np.isfinite(temperature)) and np.min(temperature[rho > 0]) > 0
    if step == 0:
        initial_support = rho > 0
    else:
        assert np.count_nonzero((rho > 0) & ~initial_support) > 0
        assert np.min(rho[rho > 0]) / np.max(rho) < 1.0e-25
print(
    "Tiny cubic tails, native charge inventory and electron positivity pass through 10 steps."
)
