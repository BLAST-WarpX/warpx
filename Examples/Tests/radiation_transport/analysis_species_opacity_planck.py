#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Validate species-aware Planck emission from deterministic electrons."""

from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer
from scipy.constants import Boltzmann, c, elementary_charge, physical_constants


def load_plotfile(path: Path) -> dict[str, np.ndarray]:
    with open(path / "Header") as header:
        header.readline()
        n_fields = int(header.readline())
        field_names = [header.readline().strip() for _ in range(n_fields)]
    return _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), field_names)


def load_table(path: Path) -> np.ndarray:
    table = np.loadtxt(path)
    return table[np.newaxis, :] if table.ndim == 1 else table


initial = load_plotfile(Path("diags/diag1000000"))
final = load_plotfile(Path("diags/diag1000001"))
radiation_table = load_table(Path("diags/radiation_energy.txt"))
particle_table = load_table(Path("diags/particle_energy.txt"))

initial_radiation = initial["radiation_diffusion_energy"].squeeze()
initial_material = initial["radiation_material_energy"].squeeze()
radiation = final["radiation_diffusion_energy"].squeeze()
material = final["radiation_material_energy"].squeeze()
assert radiation.shape == (4,)
assert radiation_table.shape == (2, 17)
assert particle_table.shape[0] == 2

dr = 1.0e-3
dt = 1.0e-10
temperature = elementary_charge / Boltzmann
radiation_constant = 4.0 * physical_constants["Stefan-Boltzmann constant"][0] / c
cell_volume = np.pi * np.array([1.0, 3.0, 5.0, 7.0]) * dr**2
alpha = np.array([0.0, 10.0, 40.0, 0.0])
expected = (
    radiation_constant * temperature**4 * cell_volume * (1.0 - np.exp(-alpha * c * dt))
)

single_precision = radiation.dtype == np.float32
analytic_rtol = 3.0e-5 if single_precision else 4.0e-12
conservation_rtol = 4.0e-5 if single_precision else 3.0e-11

np.testing.assert_array_equal(initial_radiation, 0.0)
np.testing.assert_allclose(radiation, expected, rtol=analytic_rtol, atol=1.0e-24)
np.testing.assert_array_equal(np.flatnonzero(radiation), np.array([1, 2]))

# ParticleEnergy columns are step, time, total, electrons, foam, tungsten,
# photons, followed by species mean energies. The electron loss is checked
# against the realized material transfer, which may differ from the request.
electron_loss = particle_table[0, 3] - particle_table[-1, 3]
expected_total = np.sum(expected)
np.testing.assert_allclose(radiation_table[-1, 4], expected_total, rtol=analytic_rtol)
np.testing.assert_allclose(radiation_table[:, 7:9], 0.0, atol=1.0e-24)
np.testing.assert_allclose(radiation_table[0, -2:], 0.0, atol=1.0e-24)
realized_material = material - initial_material
active_opacity_cells = np.array([1, 2])
np.testing.assert_array_equal(np.flatnonzero(realized_material), active_opacity_cells)
assert np.all(realized_material[active_opacity_cells] < 0.0)
realized_material_total = np.sum(np.asarray(realized_material, dtype=np.float64))
radiation_gain = np.sum(
    np.asarray(radiation, dtype=np.float64)
    - np.asarray(initial_radiation, dtype=np.float64)
)
requested_material = -radiation_gain
residual = radiation_table[-1, -2]
precision = np.float32 if single_precision else np.float64
field_roundoff = (
    32.0
    * np.finfo(precision).eps
    * max(
        np.max(np.abs(initial_radiation)),
        np.max(np.abs(radiation)),
        np.max(np.abs(realized_material)),
        abs(requested_material),
    )
)
np.testing.assert_allclose(
    residual,
    requested_material - realized_material_total,
    rtol=0.0,
    atol=max(1.0e-24, field_roundoff),
)
np.testing.assert_allclose(radiation_table[-1, -1], residual, rtol=0.0, atol=1.0e-24)
np.testing.assert_allclose(
    radiation_table[-1, 5],
    realized_material_total,
    rtol=0.0,
    atol=max(1.0e-24, field_roundoff),
)
np.testing.assert_allclose(
    radiation_table[-1, 6],
    realized_material_total,
    rtol=0.0,
    atol=max(1.0e-24, field_roundoff),
)
np.testing.assert_allclose(
    radiation_table[-1, 9],
    realized_material_total,
    rtol=0.0,
    atol=max(1.0e-24, field_roundoff),
)
np.testing.assert_allclose(
    electron_loss,
    -realized_material_total,
    rtol=conservation_rtol,
    atol=max(1.0e-24, field_roundoff),
)
electron_radiation_drift = (
    particle_table[-1, 3]
    - particle_table[0, 3]
    + radiation_table[-1, 2]
    - radiation_table[0, 2]
)
particle_roundoff = (
    32.0
    * np.finfo(precision).eps
    * max(
        np.max(np.abs(particle_table[:, 3])),
        np.max(np.abs(radiation_table[:, 2])),
        abs(realized_material_total),
    )
)
np.testing.assert_allclose(
    electron_radiation_drift,
    -residual,
    rtol=0.0,
    atol=max(1.0e-24, particle_roundoff),
)
if single_precision:
    assert np.isfinite(residual)
    assert residual != 0.0

print(
    "RCYL species Planck opacity: "
    f"foam={radiation[1]:.16e} J, "
    f"tungsten={radiation[2]:.16e} J, "
    f"analytic residual={np.max(np.abs(radiation - expected)):.6e} J, "
    f"kinetic residual={residual:.16e} J"
)
