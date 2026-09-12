#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse
from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer
from scipy.constants import Boltzmann, c, elementary_charge, physical_constants
from scipy.integrate import solve_ivp

parser = argparse.ArgumentParser()
parser.add_argument("--geometry", choices=["1d", "2d", "rcyl"], required=True)
args = parser.parse_args()

plotfile = Path("diags/diag000050")
with open(plotfile / "Header") as header:
    header.readline()
    n_fields = int(header.readline())
    field_names = [header.readline().strip() for _ in range(n_fields)]

fields = _read_buffer(str(plotfile), str(plotfile / "Level_0" / "Cell_H"), field_names)
radiation_field = np.squeeze(fields["radiation_diffusion_energy"])
material_field = np.squeeze(fields["radiation_material_energy"])
temperature_field = np.squeeze(fields["Te"])
pressure_field = np.squeeze(fields["Pe"])
charge_density_field = np.squeeze(fields["rho"])
single_precision = radiation_field.dtype.itemsize == 4
ledger_rtol = 2.0e-5 if single_precision else 2.0e-10

if args.geometry == "1d":
    volumes = np.full(64, 0.01 / 64.0)
    active = np.ones(64, dtype=bool)
elif args.geometry == "2d":
    volumes = np.full((32, 32), (0.01 / 32.0) ** 2)
    active = np.ones((32, 32), dtype=bool)
else:
    edges = np.linspace(0.0, 0.01, 257)
    centers = 0.5 * (edges[:-1] + edges[1:])
    volumes = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    active = (centers >= 0.0025) & (centers < 0.0075)

radiation_table = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
time = radiation_table[:, 1]
simulated_energy = radiation_table[:, 4]
cumulative_material = radiation_table[:, 6]
total_volume = np.sum(volumes[active])
simulated_density = simulated_energy / total_volume

electron_density = 1.0e24
initial_temperature = 10.0 * elementary_charge / Boltzmann
gamma = 5.0 / 3.0
alpha_planck = 1.0e3
radiation_constant = 4.0 * physical_constants["Stefan-Boltzmann constant"][0] / c
heat_capacity = electron_density * Boltzmann / (gamma - 1.0)
initial_material_density = heat_capacity * initial_temperature


def rhs(_time, radiation_density):
    temperature = np.maximum(
        (initial_material_density - radiation_density[0]) / heat_capacity, 0.0
    )
    return np.array(
        [
            c
            * alpha_planck
            * (radiation_constant * temperature**4 - radiation_density[0])
        ]
    )


reference = solve_ivp(
    rhs,
    (time[0], time[-1]),
    np.array([0.0]),
    t_eval=time,
    rtol=2.0e-12,
    atol=1.0e-12,
    method="DOP853",
)
assert reference.success
analytic_density = reference.y[0]
analytic_temperature = (initial_material_density - analytic_density) / heat_capacity

final_density_field = radiation_field / volumes
volume_weighted_temperature = (
    np.sum(temperature_field[active] * volumes[active]) / total_volume
)
spatial_nonuniformity = np.std(final_density_field[active]) / np.mean(
    final_density_field[active]
)
history_error = np.max(np.abs(simulated_density - analytic_density)) / np.max(
    analytic_density
)
final_radiation_error = (
    abs(simulated_density[-1] - analytic_density[-1]) / analytic_density[-1]
)
final_temperature_error = (
    abs(volume_weighted_temperature - analytic_temperature[-1])
    / analytic_temperature[-1]
)
conservation_error = (
    abs(simulated_energy[-1] + cumulative_material[-1]) / simulated_energy[-1]
)
current_material_error = abs(np.sum(material_field) - radiation_table[-1, 5]) / abs(
    radiation_table[-1, 5]
)
final_material_energy = np.sum(pressure_field * volumes) / (gamma - 1.0)
initial_pressure_field = (
    np.maximum(charge_density_field / elementary_charge, 0.01 * electron_density)
    * Boltzmann
    * initial_temperature
)
initial_material_energy = np.sum(initial_pressure_field * volumes) / (gamma - 1.0)
thermodynamic_ledger_error = (
    abs(final_material_energy - initial_material_energy - cumulative_material[-1])
    / simulated_energy[-1]
)

print(f"geometry:                         {args.geometry}")
print(f"analytic final radiation energy: {analytic_density[-1] * total_volume:.16e} J")
print(f"simulated final radiation energy:{simulated_energy[-1]:.16e} J")
print(f"relative history error:          {history_error:.16e}")
print(f"relative final radiation error:  {final_radiation_error:.16e}")
print(f"relative final temperature error:{final_temperature_error:.16e}")
print(f"relative spatial nonuniformity:  {spatial_nonuniformity:.16e}")
print(f"relative conservation error:     {conservation_error:.16e}")
print(f"thermodynamic ledger error:      {thermodynamic_ledger_error:.16e}")

np.testing.assert_allclose(
    simulated_density,
    analytic_density,
    rtol=3.0e-2,
    atol=np.max(analytic_density) * ledger_rtol,
)
np.testing.assert_allclose(
    simulated_energy[-1], analytic_density[-1] * total_volume, rtol=2.0e-2
)
np.testing.assert_allclose(
    volume_weighted_temperature, analytic_temperature[-1], rtol=2.0e-2
)
np.testing.assert_allclose(
    simulated_energy[-1], np.sum(radiation_field), rtol=ledger_rtol
)
np.testing.assert_allclose(
    simulated_energy[-1], -cumulative_material[-1], rtol=ledger_rtol
)
np.testing.assert_allclose(
    np.sum(material_field), radiation_table[-1, 5], rtol=ledger_rtol
)
assert spatial_nonuniformity < 3.0e-2
assert history_error < 3.0e-2
assert final_radiation_error < 2.0e-2
assert final_temperature_error < 2.0e-2
assert conservation_error < ledger_rtol
assert current_material_error < ledger_rtol
assert thermodynamic_ledger_error < 3.0e-2
