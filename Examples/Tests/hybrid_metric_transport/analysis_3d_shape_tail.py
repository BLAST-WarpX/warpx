#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Independent tensor-product cubic charge oracle for native 3D tail transport."""

import sys
from pathlib import Path

import numpy as np
import yt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "radiation_transport"))
from analysis_moving_moment_interface import nodes  # noqa: E402

yt.set_log_level(50)
cells = np.array([16, 16, 16])
dx = 0.001 / 16
initial_support = None


def cubic(position):
    coordinate = position / dx
    cell = int(np.floor(coordinate))
    f = coordinate - cell
    result = np.zeros(16)
    result[cell - 1 : cell + 3] = (
        (1 - f) ** 3 / 6,
        2 / 3 - f * f * (1 - f / 2),
        2 / 3 - (1 - f) ** 2 * (1 - (1 - f) / 2),
        f**3 / 6,
    )
    return result


for step in (0, 10):
    plot = Path(f"diags/plt{step:06d}")
    dataset = yt.load(str(plot))
    data = dataset.all_data()
    assert data["ions", "particle_weight"].size == 1
    position = [float(data["ions", f"particle_position_{axis}"][0]) for axis in "xyz"]
    charge = float(data["ions", "particle_weight"][0]) * 1.602176634e-19
    rho = nodes(plot / "raw_fields/Level_0/rho_fp_H", cells, periodic=True)
    sx, sy, sz = [cubic(value) for value in position]
    expected = charge * sx[:, None, None] * sy[None, :, None] * sz[None, None, :]
    np.testing.assert_allclose(rho * dx**3, expected, rtol=1.0e-7, atol=0)
    np.testing.assert_allclose(np.sum(rho) * dx**3, charge, rtol=1.0e-12)
    temperature = nodes(
        Path(f"diags/chk{step:06d}/Level_0/hybrid_electron_temperature_fp[level=0]_H"),
        cells,
        periodic=True,
    )
    assert np.isfinite(temperature).all() and np.min(temperature[rho > 0]) > 0
    if step == 0:
        initial_support = rho > 0
    else:
        assert np.count_nonzero((rho > 0) & ~initial_support) > 0
        assert np.min(rho[rho > 0]) / np.max(rho) < 1.0e-40
print(
    "3D tiny cubic tails: native continuity, analytic charge and positive temperature pass"
)
