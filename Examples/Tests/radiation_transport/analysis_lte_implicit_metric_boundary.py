#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check constant-CV implicit LTE exchange in cylindrical metrics."""

import argparse
from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer
from scipy.constants import Boltzmann, c, elementary_charge, physical_constants
from scipy.optimize import brentq


def load_plotfile(path: Path) -> dict[str, np.ndarray]:
    with path.joinpath("Header").open() as header:
        header.readline()
        num_fields = int(header.readline())
        names = [header.readline().strip() for _ in range(num_fields)]
    values = _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), names)
    return {
        name: np.asarray(value, dtype=np.float64).squeeze()
        for name, value in values.items()
    }


parser = argparse.ArgumentParser()
parser.add_argument("--geometry", choices=["rcylinder", "rz"], required=True)
parser.add_argument("--precision", choices=["SINGLE", "DOUBLE"], required=True)
parser.add_argument("--alpha-planck", type=float, required=True)
args = parser.parse_args()

initial = load_plotfile(Path("diags/diag000001"))
final = load_plotfile(Path("diags/diag000002"))
radiation_table = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))

if args.geometry == "rcylinder":
    radial_edges = np.arange(9, dtype=np.float64) * 1.0e-3
    volumes = np.pi * np.diff(radial_edges**2)
    expected_shape = (8,)
else:
    radial_edges = np.arange(9, dtype=np.float64) * 1.0e-3
    volumes = np.pi * np.diff(radial_edges**2)[:, np.newaxis] * 1.0e-3
    volumes = np.broadcast_to(volumes, (8, 8))
    expected_shape = (8, 8)

field_names = (
    "rho",
    "Te",
    "Pe",
    "radiation_material_energy",
    "radiation_diffusion_energy",
)
for fields in (initial, final):
    for name in field_names:
        assert fields[name].shape == expected_shape
        assert np.all(np.isfinite(fields[name]))

assert radiation_table.shape[0] == 3
assert radiation_table.shape[1] >= 17

if args.precision == "SINGLE":
    state_rtol = 5.0e-5
    ledger_rtol = 5.0e-6
    roundoff_eps = np.finfo(np.float32).eps
else:
    state_rtol = 3.0e-8
    ledger_rtol = 5.0e-10
    roundoff_eps = np.finfo(np.float64).eps

roundoff_atol = 64.0 * roundoff_eps


def scaled_roundoff(values) -> float:
    scale = max(np.max(np.abs(values)), np.finfo(np.float64).tiny)
    return max(roundoff_atol * scale, np.finfo(np.float64).tiny)


def assert_zero(values: np.ndarray, atol: float | None = None) -> None:
    if atol is None:
        atol = scaled_roundoff(values)
    np.testing.assert_allclose(values, 0.0, rtol=0.0, atol=atol)


ledger_atol = scaled_roundoff(radiation_table)

n0 = 1.0e20
minimum_radiation_density = 0.05 * n0
raw_density = 0.25 * n0
gamma = 5.0 / 3.0
temperature_0 = elementary_charge / Boltzmann
dt = 1.0e-10
radiation_constant = 4.0 * physical_constants["Stefan-Boltzmann constant"][0] / c
heat_capacity_density = n0 * Boltzmann / (gamma - 1.0)

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

# Cylindrical particle deposition is not used as an exact per-cell density
# oracle.  The radial placement rule should nevertheless keep every cell in
# the radiation-participating interval and preserve the volume-weighted raw
# density near the requested quarter-density value.
for number_density in (number_density_initial, number_density_final):
    assert np.all(number_density > minimum_radiation_density)
    assert np.all(number_density < 0.5 * n0)
np.testing.assert_allclose(
    np.sum(number_density_initial * volumes) / np.sum(volumes),
    raw_density,
    rtol=0.2,
)
np.testing.assert_allclose(
    number_density_final, number_density_initial, rtol=state_rtol
)

np.testing.assert_allclose(temperature_initial, temperature_0, rtol=state_rtol)
np.testing.assert_allclose(
    pressure_initial / ((gamma - 1.0) * temperature_initial),
    heat_capacity_density,
    rtol=state_rtol,
)
assert_zero(radiation_initial, ledger_atol)
assert_zero(material_initial, ledger_atol)

# The reduced diagnostic keeps a few non-source columns (for example the
# total material energy) as a baseline.  These are the columns owned by the
# radiation exchange and the columns that must remain unrelated to this test.
ledger_columns = np.array([4, 5, 6, 9])
unrelated_columns = np.array([3, 7, 8, 10, 11, 12, 13, 14])
assert_zero(radiation_table[:, unrelated_columns], ledger_atol)

initial_internal_energy = pressure_initial / (gamma - 1.0)
final_internal_energy = pressure_final / (gamma - 1.0)
electron_energy_change = np.sum(
    (final_internal_energy - initial_internal_energy) * volumes
)
realized_material_cells = (final_internal_energy - initial_internal_energy) * volumes

if args.alpha_planck == 0.0:
    np.testing.assert_allclose(temperature_final, temperature_initial, rtol=state_rtol)
    np.testing.assert_allclose(pressure_final, pressure_initial, rtol=state_rtol)
    assert_zero(radiation_final, ledger_atol)
    assert_zero(material_final, ledger_atol)
    np.testing.assert_allclose(
        electron_energy_change,
        0.0,
        rtol=0.0,
        atol=scaled_roundoff(initial_internal_energy * volumes),
    )
    assert_zero(radiation_table[:, ledger_columns], ledger_atol)
    np.testing.assert_allclose(
        radiation_table[:, 2], radiation_table[0, 2], rtol=state_rtol, atol=ledger_atol
    )
    np.testing.assert_array_equal(radiation_final, 0.0)
    np.testing.assert_array_equal(material_final, 0.0)
    np.testing.assert_array_equal(radiation_table[:, ledger_columns], 0.0)
    np.testing.assert_array_equal(radiation_table[:, -2:], 0.0)
    print(f"{args.geometry} zero-opacity metric control: state and ledgers unchanged")
    raise SystemExit(0)

exchange_fraction = -np.expm1(-args.alpha_planck * c * dt)


def implicit_residual(temperature: float) -> float:
    return (
        heat_capacity_density * (temperature - temperature_0)
        + exchange_fraction * radiation_constant * temperature**4
    )


temperature_1 = brentq(implicit_residual, 0.0, temperature_0, xtol=1.0e-12)
radiation_density = exchange_fraction * radiation_constant * temperature_1**4
expected_radiation = radiation_density * volumes
expected_total = np.sum(expected_radiation)

print(f"geometry:                    {args.geometry}")
print(f"correct implicit T1:         {temperature_1:.16e} K")
print(f"measured cell Te:            {temperature_final}")
print(f"expected radiation energy:   {expected_total:.16e} J")
print(f"measured radiation energy:   {np.sum(radiation_final):.16e} J")
print(f"Pe-derived electron change:  {electron_energy_change:.16e} J")

np.testing.assert_allclose(temperature_final, temperature_1, rtol=state_rtol)
np.testing.assert_allclose(
    pressure_final / ((gamma - 1.0) * temperature_final),
    heat_capacity_density,
    rtol=state_rtol,
)
np.testing.assert_allclose(
    radiation_final, expected_radiation, rtol=ledger_rtol, atol=ledger_atol
)
np.testing.assert_allclose(
    material_final,
    realized_material_cells,
    rtol=state_rtol,
    atol=scaled_roundoff(initial_internal_energy * volumes),
)
np.testing.assert_allclose(
    electron_energy_change,
    -expected_total,
    rtol=state_rtol,
    atol=scaled_roundoff(expected_total),
)

np.testing.assert_allclose(
    radiation_table[1, ledger_columns], 0.0, rtol=0.0, atol=ledger_atol
)
realized_material = np.sum(realized_material_cells)
# q is the measured radiation-to-material transfer.  Keep the analytic LTE
# result above as an independent radiation check; do not let its solver/float
# error contaminate the q-d residual oracle.
requested_material = np.sum(radiation_initial) - np.sum(radiation_final)
residual = radiation_table[-1, -2]
cumulative_residual = radiation_table[-1, -1]
field_tolerance = max(ledger_atol, scaled_roundoff(expected_total))
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
    radiation_table[-1, ledger_columns],
    np.array([expected_total, realized_material, realized_material, realized_material]),
    rtol=ledger_rtol,
    atol=ledger_atol,
)
np.testing.assert_allclose(
    radiation_table[-1, 4], np.sum(radiation_final), rtol=ledger_rtol, atol=ledger_atol
)
np.testing.assert_allclose(
    radiation_table[-1, 5], np.sum(material_final), rtol=ledger_rtol, atol=ledger_atol
)
np.testing.assert_allclose(
    electron_energy_change + radiation_table[-1, 2] - radiation_table[1, 2],
    -residual,
    rtol=state_rtol,
    atol=field_tolerance,
)
