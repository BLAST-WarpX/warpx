#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Analytic two-group LTE emission for registry-selected HDF5 tables."""

import argparse
from pathlib import Path

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from read_raw_data import _read_buffer
from scipy.constants import Boltzmann, c, physical_constants
from scipy.integrate import quad


def load_plotfile(path: Path) -> dict[str, np.ndarray]:
    with path.joinpath("Header").open() as header:
        header.readline()
        n_fields = int(header.readline())
        names = [header.readline().strip() for _ in range(n_fields)]
    values = _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), names)
    return {
        name: np.asarray(value, dtype=np.float64).squeeze()
        for name, value in values.items()
    }


parser = argparse.ArgumentParser()
add_precision_arguments(parser)
args = parser.parse_args()
_, _, cross_dtype = precision_dtypes(args)

initial = load_plotfile(Path("diags/diag000000"))
final = load_plotfile(Path("diags/diag000001"))
radiation_table = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))

expected_shape = (40,)
field_names = (
    "radiation_material_energy",
    "radiation_diffusion_energy_g0",
    "radiation_diffusion_energy_g1",
)
for fields in (initial, final):
    for name in field_names:
        assert fields[name].shape == expected_shape
        assert np.all(np.isfinite(fields[name]))

assert radiation_table.shape[0] == 2
assert radiation_table.shape[1] >= 19
assert np.all(np.isfinite(radiation_table))

if cross_dtype == np.float32:
    cross_analytic_rtol = 3.0e-5
else:
    cross_analytic_rtol = 3.0e-12
cross_ledger_eps = np.finfo(cross_dtype).eps

cell_volume = 0.025
temperature = 5.0e4
mass_density = 10.0
dt = 0.775 / c
edge = 2.0e-19
temperature_energy = Boltzmann * temperature
planck_integral = np.pi**4 / 15.0


def planck_integrand(x: float) -> float:
    return 0.0 if x == 0.0 else x**3 / np.expm1(x)


edge_ratio = edge / temperature_energy
low_fraction = quad(planck_integrand, 0.0, edge_ratio)[0] / planck_integral
planck_fractions = np.array([low_fraction, 1.0 - low_fraction])
radiation_constant = 4.0 * physical_constants["Stefan-Boltzmann constant"][0] / c
equilibrium_group_energy = (
    radiation_constant * temperature**4 * cell_volume * planck_fractions
)

# The fixture's mass opacities are constant in density and temperature.  The
# resulting linear coefficients are the independent per-material/group oracle.
linear_coefficients = (
    np.array(
        [
            [3.162277660168379e-2, 3.162277660168379e-1],
            [6.324555320336758e-2, 6.324555320336758e-1],
        ]
    )
    * mass_density
)
relaxation = -np.expm1(-linear_coefficients * c * dt)
emission = relaxation * equilibrium_group_energy[np.newaxis, :]

active_cells = [*range(8, 16), *range(24, 32)]
inactive_cells = np.setdiff1d(np.arange(40), active_cells)
interior_slices = (slice(9, 15), slice(25, 31))
interface_pairs = ((8, 15), (24, 31))

group_zero = final["radiation_diffusion_energy_g0"]
group_one = final["radiation_diffusion_energy_g1"]
measured_groups = np.column_stack((group_zero, group_one))
initial_material = initial["radiation_material_energy"]
final_material = final["radiation_material_energy"]
realized_material = final_material - initial_material

assert np.flatnonzero(group_zero).tolist() == active_cells
assert np.flatnonzero(group_one).tolist() == active_cells
np.testing.assert_array_equal(measured_groups[inactive_cells], 0.0)
for material, (interior, interfaces) in enumerate(
    zip(interior_slices, interface_pairs)
):
    np.testing.assert_allclose(
        measured_groups[interior],
        np.broadcast_to(emission[material], measured_groups[interior].shape),
        rtol=cross_analytic_rtol,
        atol=1.0e-24,
    )
    np.testing.assert_allclose(
        measured_groups[interfaces[0]],
        measured_groups[interfaces[1]],
        rtol=cross_analytic_rtol,
        atol=1.0e-24,
    )
    assert np.all(measured_groups[list(interfaces)] > 0.0)
    assert np.all(measured_groups[list(interfaces)] < emission[material])

# Requested LTE energy is cell-local, while the authoritative realized hybrid
# ledger measures the nodal EOS remap and therefore spreads across cells next
# to each material interface.  Preserve that spatial distinction and use the
# signed residual below for exact global closure.
expected_material_support = [*range(7, 17), *range(23, 33)]
assert np.flatnonzero(realized_material).tolist() == expected_material_support
assert np.all(realized_material[expected_material_support] < 0.0)

realized_total = np.sum(realized_material)
measured_total = np.sum(measured_groups)
print(f"Planck edge / kBT:             {edge_ratio:.16e}")
print(f"Planck group fractions:         {planck_fractions}")
print(f"analytic interior emission:     {emission}")
print(f"measured total radiation:       {measured_total:.16e} J")
print(f"realized material transfer:     {realized_total:.16e} J")

# RadiationEnergy columns are [step, time, total, streaming, diffusion,
# material current, material cumulative, boundary current, boundary
# cumulative, group-0, group-1, internal material, kinetic material,
# streaming-boundary, cumulative streaming-boundary, diffusion-boundary,
# cumulative diffusion-boundary, residual, cumulative residual].
np.testing.assert_allclose(
    radiation_table[-1, 2], measured_total, rtol=cross_analytic_rtol
)
np.testing.assert_allclose(
    radiation_table[-1, 4], measured_total, rtol=cross_analytic_rtol
)
np.testing.assert_allclose(
    radiation_table[-1, 9], np.sum(group_zero), rtol=cross_analytic_rtol
)
np.testing.assert_allclose(
    radiation_table[-1, 10], np.sum(group_one), rtol=cross_analytic_rtol
)

roundoff_scale = max(
    np.max(np.abs(radiation_table)),
    abs(realized_total),
    abs(measured_total),
    np.finfo(np.float64).tiny,
)
ledger_atol = max(1.0e-24, 64.0 * cross_ledger_eps * roundoff_scale)
np.testing.assert_allclose(radiation_table[:, 3], 0.0, rtol=0.0, atol=ledger_atol)
np.testing.assert_allclose(radiation_table[:, 7:9], 0.0, rtol=0.0, atol=ledger_atol)
np.testing.assert_allclose(radiation_table[:, 12:17], 0.0, rtol=0.0, atol=ledger_atol)
np.testing.assert_allclose(
    radiation_table[-1, 5], realized_total, rtol=0.0, atol=ledger_atol
)
np.testing.assert_allclose(
    radiation_table[-1, 6], realized_total, rtol=0.0, atol=ledger_atol
)
np.testing.assert_allclose(
    radiation_table[-1, 11], realized_total, rtol=0.0, atol=ledger_atol
)
closure_residual = radiation_table[0, 2] - radiation_table[-1, 2] - realized_total
np.testing.assert_allclose(
    radiation_table[-1, -2], closure_residual, rtol=0.0, atol=ledger_atol
)
np.testing.assert_allclose(
    radiation_table[-1, -1], radiation_table[-1, -2], rtol=0.0, atol=ledger_atol
)
