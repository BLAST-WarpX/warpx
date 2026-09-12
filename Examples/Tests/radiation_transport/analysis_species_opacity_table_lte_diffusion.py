#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Validate mixed 2D/3D species Planck and Rosseland opacity tables."""

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
radiation = load_table(Path("diags/radiation_energy.txt"))
particle = load_table(Path("diags/particle_energy.txt"))
momentum = load_table(Path("diags/particle_momentum.txt"))
number = load_table(Path("diags/particle_number.txt"))

initial_diffusion = initial["radiation_diffusion_energy"].squeeze()
initial_material = initial["radiation_material_energy"].squeeze()
diffusion = final["radiation_diffusion_energy"].squeeze()
material = final["radiation_material_energy"].squeeze()
assert diffusion.shape == (4,)
assert radiation.shape[0] == 2
assert radiation.shape[1] >= 17
assert particle.shape[0] == 2
assert particle.shape[1] >= 12

dr = 1.0e-3
dt = 1.0e-13
temperature = elementary_charge / Boltzmann
radiation_constant = 4.0 * physical_constants["Stefan-Boltzmann constant"][0] / c
source_volume = 5.0 * np.pi * dr**2

# At ni=1.25*n0, Te=1 eV and representative energy=100 eV:
#   Planck:    foam 2D = 1125, tungsten 3D = 4250 -> 5375 m^-1
#   Rosseland: foam 3D = 425,   tungsten 2D = 750   -> 1175 m^-1
alpha_planck = 1125.0 + 4250.0
alpha_rosseland = 425.0 + 750.0
emitted = (
    radiation_constant
    * temperature**4
    * source_volume
    * (1.0 - np.exp(-alpha_planck * c * dt))
)

# The configured CFL bound requests exactly one explicit diffusion substep.
requested_substeps = np.ceil(4.0 * c * dt / (3.0 * 1.0 * 0.1 * dr))
assert requested_substeps == 1.0

# Only cell 2 has material opacity. At each adjacent face the arithmetic
# Rosseland coefficient is alpha_R/2. The FLD stencil therefore has
# R=|grad E|/(alpha_face*E_face)=4/(alpha_R*dr).
face_opacity = 0.5 * alpha_rosseland
dimensionless_gradient = 2.0 / (face_opacity * dr)
flux_limiter = (2.0 + dimensionless_gradient) / (
    6.0 + 3.0 * dimensionless_gradient + dimensionless_gradient**2
)
diffusion_coefficient = c * flux_limiter / face_opacity
source_density = emitted / source_volume
inner_transfer = (
    diffusion_coefficient * (2.0 * np.pi * 2.0 * dr) * source_density / dr * dt
)
outer_transfer = (
    diffusion_coefficient * (2.0 * np.pi * 3.0 * dr) * source_density / dr * dt
)
expected_cells = np.array(
    [0.0, inner_transfer, emitted - inner_transfer - outer_transfer, outer_transfer]
)

single_precision = diffusion.dtype == np.float32
analytic_rtol = 5.0e-5 if single_precision else 7.0e-12
conservation_rtol = 7.0e-5 if single_precision else 4.0e-11

np.testing.assert_array_equal(initial_diffusion, 0.0)
np.testing.assert_allclose(diffusion, expected_cells, rtol=analytic_rtol, atol=1.0e-24)
np.testing.assert_allclose(np.sum(diffusion), emitted, rtol=analytic_rtol)

# Diffusion only redistributes the emitted LTE energy; material realization and
# the electron loss are checked against the signed realized transfer.
electron_loss = particle[0, 3] - particle[-1, 3]
np.testing.assert_allclose(radiation[-1, 4], emitted, rtol=analytic_rtol)
np.testing.assert_allclose(radiation[:, 7:9], 0.0, atol=1.0e-24)
np.testing.assert_allclose(radiation[0, -2:], 0.0, atol=1.0e-24)
realized_material = material - initial_material
emitting_material_cells = np.array([2])
np.testing.assert_array_equal(
    np.flatnonzero(realized_material), emitting_material_cells
)
assert np.all(realized_material[emitting_material_cells] < 0.0)
realized_material_total = np.sum(np.asarray(realized_material, dtype=np.float64))
radiation_gain = np.sum(
    np.asarray(diffusion, dtype=np.float64)
    - np.asarray(initial_diffusion, dtype=np.float64)
)
requested_material = -radiation_gain
residual = radiation[-1, -2]
precision = np.float32 if single_precision else np.float64
field_roundoff = (
    32.0
    * np.finfo(precision).eps
    * max(
        np.max(np.abs(initial_diffusion)),
        np.max(np.abs(diffusion)),
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
np.testing.assert_allclose(radiation[-1, -1], residual, rtol=0.0, atol=1.0e-24)
np.testing.assert_allclose(
    radiation[-1, 5],
    realized_material_total,
    rtol=0.0,
    atol=max(1.0e-24, field_roundoff),
)
np.testing.assert_allclose(
    radiation[-1, 6],
    realized_material_total,
    rtol=0.0,
    atol=max(1.0e-24, field_roundoff),
)
np.testing.assert_allclose(
    radiation[-1, 9],
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
    particle[-1, 3] - particle[0, 3] + radiation[-1, 2] - radiation[0, 2]
)
particle_roundoff = (
    32.0
    * np.finfo(precision).eps
    * max(
        np.max(np.abs(particle[:, 3])),
        np.max(np.abs(radiation[:, 2])),
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

# The symmetric dummy photons and paired electrons preserve exact zero total
# momentum. No species population changes in an LTE opacity lookup.
momentum_scale = max(radiation[0, 2] / c, emitted / c)
np.testing.assert_allclose(
    momentum[:, 2:5], 0.0, atol=conservation_rtol * momentum_scale
)
initial_number = np.broadcast_to(number[0], number.shape)
np.testing.assert_array_equal(number[:, 3:7], initial_number[:, 3:7])
np.testing.assert_allclose(number[:, 8:12], initial_number[:, 8:12], rtol=analytic_rtol)

print(
    "RCYL mixed species Planck/Rosseland tables: "
    f"emitted={emitted:.16e} J, "
    f"inner={diffusion[1]:.16e} J, outer={diffusion[3]:.16e} J, "
    f"kinetic residual={residual:.16e} J, "
    f"electron+radiation drift={electron_radiation_drift:.6e} J"
)
