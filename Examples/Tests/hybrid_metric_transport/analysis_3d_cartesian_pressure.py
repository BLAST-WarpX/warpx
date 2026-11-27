#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Independent native EOS/coordinate check, not an evolved flow test."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "radiation_transport"))
from analysis_moving_moment_interface import nodes  # noqa: E402

cells = np.array([16, 16, 16])
periodic = [False, False, False]
rho = nodes(
    Path("diags/plt000000/raw_fields/Level_0/rho_fp_H"), cells, periodic=periodic
)
temperature = nodes(
    Path("diags/chk000000/Level_0/hybrid_electron_temperature_fp[level=0]_H"),
    cells,
    periodic=periodic,
)
assert rho.shape == (17, 17, 17) and temperature.shape == rho.shape
assert np.all(np.isfinite(rho)) and np.all(rho > 0)
assert np.all(np.isfinite(temperature)) and np.all(temperature > 0)
assert np.max(rho) / np.min(rho) > 20
fraction = np.linspace(0.0, 1.0, 17)
expected = 1.0e6 * (
    1
    + 0.05 * fraction[:, None, None]
    + 0.07 * fraction[None, :, None]
    + 0.09 * fraction[None, None, :]
)
pressure = rho * 1.380649e-23 * temperature / 1.602176634e-19
np.testing.assert_allclose(pressure, expected, rtol=256 * np.finfo(float).eps, atol=0)
print(
    f"Native 3D deposited-pressure maximum relative error: {np.max(np.abs(pressure / expected - 1)):.4e}"
)
