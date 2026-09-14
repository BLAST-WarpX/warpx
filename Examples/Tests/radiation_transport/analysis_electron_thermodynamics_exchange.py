#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Validate EOS-derived hybrid LTE exchange and its conservative RCYL remap."""

import argparse
from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer
from scipy.constants import Boltzmann, c, elementary_charge
from scipy.optimize import brentq


def load_plotfile(path: Path) -> dict[str, np.ndarray]:
    with open(path / "Header") as header:
        header.readline()
        n_fields = int(header.readline())
        field_names = [header.readline().strip() for _ in range(n_fields)]
    fields = _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), field_names)
    return {
        name: np.asarray(value, dtype=np.float64).squeeze()
        for name, value in fields.items()
    }


parser = argparse.ArgumentParser()
parser.add_argument("--precision", choices=["SINGLE", "DOUBLE"], default="DOUBLE")
args = parser.parse_args()

if args.precision == "SINGLE":
    eos_rtol = 3.0e-5
    analytic_rtol = 4.0e-5
    ledger_rtol = 5.0e-5
    remap_rtol = 3.0e-4
else:
    eos_rtol = 3.0e-11
    analytic_rtol = 3.0e-9
    ledger_rtol = 3.0e-10
    # Pe/Te are inspected after the QDSMC part of the hybrid step.  The
    # cell-integrated radiation/material ledgers below retain tighter bounds.
    remap_rtol = 2.0e-8

# The iteration-0 plot precedes hybrid state initialization.  Step 1 is the
# initialized state after the zero-opacity warm-up; step 2 follows one active
# LTE exchange.
initial = load_plotfile(Path("diags/diag000001"))
final = load_plotfile(Path("diags/diag000002"))
radiation_table = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))

n_cell = 32
radius = 1.0e-2
edges = np.linspace(0.0, radius, n_cell + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
volumes = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
active = (centers > 2.5e-3) & (centers < 7.5e-3)
core = active.copy()
halo = active.copy()
for offset in (1, 2):
    core[:-offset] &= active[offset:]
    core[offset:] &= active[:-offset]
    halo[:-offset] |= active[offset:]
    halo[offset:] |= active[:-offset]

for fields in (initial, final):
    for name in (
        "rho",
        "Te",
        "Pe",
        "radiation_material_energy",
        "radiation_diffusion_energy",
    ):
        assert fields[name].shape == (n_cell,)

gamma = 5.0 / 3.0
n_floor = 1.0e24
raw_density = 0.25 * n_floor
minimum_radiation_density = 0.05 * n_floor
temperature_0 = 10.0 * elementary_charge / Boltzmann
alpha_planck = 5.0e3
dt = 4.0e-13
radiation_constant = 7.565733250280007e-16
heat_capacity_floor = n_floor * Boltzmann / (gamma - 1.0)

rho_0 = initial["rho"]
rho_1 = final["rho"]
ne_0 = rho_0 / elementary_charge
ne_1 = rho_1 / elementary_charge
temperature_initial = initial["Te"]
temperature_final = final["Te"]
pressure_initial = initial["Pe"]
pressure_final = final["Pe"]
radiation_initial = initial["radiation_diffusion_energy"]
radiation_final = final["radiation_diffusion_energy"]
material_initial = initial["radiation_material_energy"]
material_final = final["radiation_material_energy"]

# The manufactured plasma must participate in radiation exchange, but its raw
# deposited density must never select the thermodynamic state.  Bounds are used
# instead of treating the ordinary rho diagnostic as an exact particle-density
# oracle in cylindrical geometry.
assert np.min(ne_0[active]) > minimum_radiation_density
assert np.max(ne_0[active]) < 0.5 * n_floor
assert 0.20 * n_floor < np.mean(ne_0[active]) < 0.30 * n_floor
np.testing.assert_allclose(ne_1[active], ne_0[active], rtol=eos_rtol)

# For the ideal backend, Pe exposes both caloric quantities without adding a
# test-only field: U_e=Pe/(gamma-1), C_Ve=U_e/Te.  Their floor-density values
# must hold before and after the radiation source is remapped to hybrid nodes.
energy_density_initial = pressure_initial / (gamma - 1.0)
energy_density_final = pressure_final / (gamma - 1.0)
heat_capacity_initial = energy_density_initial / temperature_initial
heat_capacity_final = energy_density_final / temperature_final
np.testing.assert_allclose(temperature_initial[halo], temperature_0, rtol=eos_rtol)
np.testing.assert_allclose(
    heat_capacity_initial[halo], heat_capacity_floor, rtol=eos_rtol
)
np.testing.assert_allclose(
    heat_capacity_final[halo], heat_capacity_floor, rtol=eos_rtol
)
np.testing.assert_allclose(
    energy_density_initial[halo],
    heat_capacity_floor * temperature_initial[halo],
    rtol=eos_rtol,
)
np.testing.assert_allclose(
    energy_density_final[halo],
    heat_capacity_floor * temperature_final[halo],
    rtol=eos_rtol,
)

# The one-group implicit solve obeys
#   Cv (T1-T0) + (1-exp(-alpha*c*dt)) a T1^4 = 0
# in every active cell.  This is deliberately evaluated with Cv at n_floor.
exchange_fraction = -np.expm1(-alpha_planck * c * dt)


def implicit_residual(temperature: float, heat_capacity: float) -> float:
    return (
        heat_capacity * (temperature - temperature_0)
        + exchange_fraction * radiation_constant * temperature**4
    )


temperature_1 = brentq(
    implicit_residual,
    0.0,
    temperature_0,
    args=(heat_capacity_floor,),
    xtol=1.0e-12,
)
radiation_density = exchange_fraction * radiation_constant * temperature_1**4
expected_radiation = np.zeros(n_cell)
expected_radiation[active] = radiation_density * volumes[active]
expected_total = np.sum(expected_radiation)

np.testing.assert_array_equal(radiation_initial, 0.0)
np.testing.assert_array_equal(material_initial, 0.0)
np.testing.assert_allclose(
    radiation_final[active], expected_radiation[active], rtol=analytic_rtol
)
np.testing.assert_array_equal(radiation_final[~active], 0.0)
np.testing.assert_allclose(temperature_final[core], temperature_1, rtol=analytic_rtol)
np.testing.assert_allclose(
    heat_capacity_floor * (temperature_final[core] - temperature_0)
    + radiation_final[core] / volumes[core],
    0.0,
    atol=analytic_rtol * radiation_density,
)

# The old raw-density reconstruction predicts a measurably different answer;
# keep the fixture discriminating even if its particle deposition is changed.
raw_heat_capacity = raw_density * Boltzmann / (gamma - 1.0)
raw_temperature_1 = brentq(
    implicit_residual,
    0.0,
    temperature_0,
    args=(raw_heat_capacity,),
    xtol=1.0e-12,
)
raw_radiation_density = exchange_fraction * radiation_constant * raw_temperature_1**4
assert abs(raw_radiation_density / radiation_density - 1.0) > 0.15

# Cell-centered Pe is the arithmetic image of the nodal state.  In RCYL its
# volume integral exactly reproduces the nodal dual-volume energy change when
# the source is away from the physical boundaries.  Core cells close locally;
# the interface cells are allowed to share energy with the two-cell QDSMC halo.
electron_change = (energy_density_final - energy_density_initial) * volumes
material_change = material_final - material_initial
np.testing.assert_allclose(
    electron_change[core], material_change[core], rtol=remap_rtol
)
realized_material = np.sum(material_change)
np.testing.assert_allclose(np.sum(electron_change), realized_material, rtol=remap_rtol)
assert np.max(np.abs(electron_change[~halo])) < remap_rtol * expected_total

# RadiationEnergy provides an independent global reduction.  The dummy photon
# packet is stationary, so only the diffusion column changes.  The requested
# transfer is q=-Delta E_radiation, while the material field and all three
# material ledgers record the EOS-realized d.  Their difference is the
# numerical residual R=q-d.
assert radiation_table.shape[0] == 3
assert radiation_table.shape[1] >= 17
np.testing.assert_allclose(radiation_table[-1, 4], expected_total, rtol=analytic_rtol)
np.testing.assert_allclose(
    radiation_table[-1, 4], np.sum(radiation_final), rtol=ledger_rtol
)
np.testing.assert_allclose(radiation_table[-1, 5], realized_material, rtol=ledger_rtol)
np.testing.assert_allclose(radiation_table[-1, 6], realized_material, rtol=ledger_rtol)
np.testing.assert_allclose(radiation_table[-1, 9], realized_material, rtol=ledger_rtol)
np.testing.assert_allclose(
    radiation_table[:, 3], radiation_table[1, 3], rtol=ledger_rtol
)
np.testing.assert_allclose(radiation_table[:, 7:9], 0.0, atol=1.0e-24)
np.testing.assert_allclose(radiation_table[:, 10:-2], 0.0, atol=1.0e-24)

# q is the measured radiation-to-material transfer.  Keep the analytic LTE
# result above as an independent radiation check; do not let its solver/float
# error contaminate the q-d residual oracle.
requested_material = np.sum(radiation_initial) - np.sum(radiation_final)
residual = radiation_table[-1, -2]
cumulative_residual = radiation_table[-1, -1]
precision = np.float32 if args.precision == "SINGLE" else np.float64
field_roundoff = (
    32.0
    * np.finfo(precision).eps
    * max(
        abs(radiation_table[0, 2]),
        abs(radiation_table[-1, 2]),
        abs(requested_material),
        abs(realized_material),
        abs(radiation_table[-1, 5]),
        abs(radiation_table[-1, 6]),
        abs(radiation_table[-1, 9]),
    )
)
field_tolerance = max(1.0e-24, field_roundoff)
closure_residual = radiation_table[0, 2] - radiation_table[-1, 2] - realized_material
np.testing.assert_allclose(residual, closure_residual, rtol=0.0, atol=field_tolerance)
np.testing.assert_allclose(
    residual,
    requested_material - realized_material,
    rtol=0.0,
    atol=field_tolerance,
)
np.testing.assert_allclose(
    cumulative_residual, residual, rtol=0.0, atol=field_tolerance
)
np.testing.assert_allclose(
    radiation_table[:2, -2:], 0.0, rtol=0.0, atol=field_tolerance
)
np.testing.assert_allclose(
    realized_material + radiation_table[-1, 2] - radiation_table[0, 2],
    -residual,
    rtol=0.0,
    atol=field_tolerance,
)

print(
    "RCYL electron-thermodynamics LTE exchange: "
    f"raw_ne/n_floor={np.mean(ne_0[active]) / n_floor:.8f}, "
    f"T1={temperature_1:.10e} K, radiation={expected_total:.10e} J, "
    f"realized material={realized_material:.10e} J, "
    f"numerical residual={residual:.6e} J"
)
