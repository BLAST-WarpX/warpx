#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check nonperiodic constant-CV implicit-LTE gather/remap consistency."""

import argparse
from pathlib import Path

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from read_raw_data import _read_buffer
from scipy.constants import Boltzmann, c, elementary_charge
from scipy.optimize import brentq


def load_plotfile(path: Path) -> dict[str, np.ndarray]:
    with (path / "Header").open() as header:
        header.readline()
        num_fields = int(header.readline())
        names = [header.readline().strip() for _ in range(num_fields)]
    values = _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), names)
    return {
        name: np.asarray(value, dtype=np.float64).squeeze()
        for name, value in values.items()
    }


parser = argparse.ArgumentParser()
parser.add_argument("--alpha-planck", type=float, required=True)
add_precision_arguments(parser)
args = parser.parse_args()
field_dtype, _, cross_dtype = precision_dtypes(args)

if field_dtype == np.float32:
    state_rtol = 5.0e-5
else:
    state_rtol = 2.0e-8
if cross_dtype == np.float32:
    cross_state_rtol = 5.0e-5
    cross_ledger_rtol = 5.0e-6
else:
    cross_state_rtol = 2.0e-8
    cross_ledger_rtol = 2.0e-11

initial = load_plotfile(Path("diags/diag000001"))
final = load_plotfile(Path("diags/diag000002"))
radiation_table = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))

field_names = (
    "rho",
    "Te",
    "Pe",
    "radiation_material_energy",
    "radiation_diffusion_energy",
)
for fields in (initial, final):
    for name in field_names:
        assert fields[name].shape == (8,)
        assert np.all(np.isfinite(fields[name]))

n0 = 1.0e20
raw_density = 0.25 * n0
minimum_radiation_density = 0.05 * n0
gamma = 5.0 / 3.0
temperature_0 = elementary_charge / Boltzmann
dt = 1.0e-10
dx = 1.0
radiation_constant = 7.565733250280007e-16
heat_capacity_density = n0 * Boltzmann / (gamma - 1.0)
cell_apply_capacity = heat_capacity_density * dx
exchange_fraction = -np.expm1(-args.alpha_planck * c * dt)

temperature_initial = initial["Te"]
temperature_final = final["Te"]
pressure_initial = initial["Pe"]
pressure_final = final["Pe"]
radiation_initial = initial["radiation_diffusion_energy"]
radiation_final = final["radiation_diffusion_energy"]
material_initial = initial["radiation_material_energy"]
material_final = final["radiation_material_energy"]
number_density_initial = initial["rho"] / elementary_charge
number_density_final = final["rho"] / elementary_charge

assert np.all(number_density_initial > minimum_radiation_density)
assert np.all(number_density_initial < 0.5 * n0)
np.testing.assert_allclose(number_density_initial, raw_density, rtol=state_rtol)
np.testing.assert_allclose(
    number_density_final, number_density_initial, rtol=state_rtol
)
np.testing.assert_allclose(temperature_initial, temperature_0, rtol=cross_state_rtol)
np.testing.assert_allclose(
    pressure_initial / ((gamma - 1.0) * temperature_initial),
    heat_capacity_density,
    rtol=state_rtol,
)
np.testing.assert_array_equal(radiation_initial, 0.0)
np.testing.assert_array_equal(material_initial, 0.0)


def residual(temperature: float, capacity: float) -> float:
    return (
        capacity * (temperature - temperature_0)
        + exchange_fraction * radiation_constant * dx * temperature**4
    )


if args.alpha_planck > 0.0:
    temperature_1 = brentq(
        residual, 0.0, temperature_0, args=(cell_apply_capacity,), xtol=1.0e-12
    )
    expected_radiation = exchange_fraction * radiation_constant * dx * temperature_1**4
    expected_temperature = np.full(8, temperature_1)

    old_gather_capacity = 0.75 * cell_apply_capacity
    old_root_temperature = brentq(
        residual,
        0.0,
        temperature_0,
        args=(old_gather_capacity,),
        xtol=1.0e-12,
    )
    old_boundary_radiation = (
        exchange_fraction * radiation_constant * dx * old_root_temperature**4
    )
    relative_kill_gap = abs(old_boundary_radiation / expected_radiation - 1.0)
    assert relative_kill_gap > 0.1
else:
    temperature_1 = temperature_0
    expected_radiation = 0.0
    expected_temperature = np.full(8, temperature_0)
    old_root_temperature = temperature_0
    old_boundary_radiation = 0.0
    relative_kill_gap = 0.0

print(f"correct implicit T1:       {temperature_1:.16e} K")
print(f"measured cell Te:         {temperature_final}")
print(f"old boundary gather root: {old_root_temperature:.16e} K")
print(f"old boundary radiation:   {old_boundary_radiation:.16e} J")
print(f"pre-fix ledger kill gap:  {relative_kill_gap:.16e}")

np.testing.assert_allclose(
    temperature_final, expected_temperature, rtol=cross_state_rtol
)
np.testing.assert_allclose(
    pressure_final / ((gamma - 1.0) * temperature_final),
    heat_capacity_density,
    rtol=state_rtol,
)
np.testing.assert_allclose(
    radiation_final,
    expected_radiation,
    rtol=cross_ledger_rtol,
    atol=np.finfo(np.float64).tiny,
)

initial_internal_energy = pressure_initial / (gamma - 1.0)
final_internal_energy = pressure_final / (gamma - 1.0)
realized_material_cells = (final_internal_energy - initial_internal_energy) * dx
electron_energy_change = np.sum((final_internal_energy - initial_internal_energy) * dx)
electron_roundoff_atol = (
    64.0 * np.finfo(field_dtype).eps * np.sum(np.abs(initial_internal_energy) * dx)
)
cross_roundoff_atol = (
    64.0 * np.finfo(cross_dtype).eps * np.sum(np.abs(initial_internal_energy) * dx)
)
np.testing.assert_allclose(
    material_final,
    realized_material_cells,
    rtol=cross_state_rtol,
    atol=max(electron_roundoff_atol, cross_roundoff_atol),
)
np.testing.assert_allclose(
    electron_energy_change,
    -8.0 * expected_radiation,
    rtol=cross_state_rtol,
    atol=max(electron_roundoff_atol, cross_roundoff_atol),
)
np.testing.assert_allclose(
    np.sum(radiation_final), 8.0 * expected_radiation, rtol=cross_ledger_rtol
)
np.testing.assert_allclose(
    np.sum(material_final),
    electron_energy_change,
    rtol=cross_ledger_rtol,
    atol=max(electron_roundoff_atol, cross_roundoff_atol),
)

assert radiation_table.shape[0] == 3
assert radiation_table.shape[1] >= 17
realized_material = np.sum(material_final)
# q is the measured radiation-to-material transfer.  Keep the analytic LTE
# result above as an independent radiation check; do not let its solver/float
# error contaminate the q-d residual oracle.
requested_material = np.sum(radiation_initial) - np.sum(radiation_final)
residual = radiation_table[-1, -2]
cumulative_residual = radiation_table[-1, -1]
field_roundoff = (
    32.0
    * np.finfo(cross_dtype).eps
    * max(
        abs(radiation_table[0, 2]),
        abs(radiation_table[-1, 2]),
        abs(requested_material),
        abs(realized_material),
        abs(radiation_table[-1, 5]),
        abs(radiation_table[-1, 6]),
    )
)
field_tolerance = max(
    1.0e-24, field_roundoff, electron_roundoff_atol, cross_roundoff_atol
)
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
    radiation_table[-1, 4], 8.0 * expected_radiation, rtol=cross_ledger_rtol
)
np.testing.assert_allclose(
    radiation_table[-1, 5], realized_material, rtol=0.0, atol=field_tolerance
)
np.testing.assert_allclose(
    radiation_table[-1, 6], realized_material, rtol=0.0, atol=field_tolerance
)
np.testing.assert_allclose(
    radiation_table[-1, 9], realized_material, rtol=0.0, atol=field_tolerance
)
np.testing.assert_allclose(radiation_table[:, 3], 0.0, atol=1.0e-30)
np.testing.assert_allclose(radiation_table[:, 7:9], 0.0, atol=1.0e-30)
np.testing.assert_allclose(radiation_table[:, 10:-2], 0.0, atol=1.0e-30)
np.testing.assert_allclose(radiation_table[:2, -2:], 0.0, atol=field_tolerance)

electron_radiation_drift = (
    electron_energy_change + radiation_table[-1, 2] - radiation_table[0, 2]
)
np.testing.assert_allclose(
    electron_radiation_drift, -residual, rtol=0.0, atol=field_tolerance
)
if args.alpha_planck == 0.0:
    np.testing.assert_array_equal(radiation_final, 0.0)
    np.testing.assert_array_equal(material_final, 0.0)
    np.testing.assert_array_equal(radiation_table[:, [2, 4, 5, 6, 9]], 0.0)
    np.testing.assert_array_equal(radiation_table[:, -2:], 0.0)

print(f"expected diffusion energy:{expected_radiation:.16e} J")
print(f"electron energy change:   {electron_energy_change:.16e} J")
print(f"material realization residual:{residual:.16e} J")
