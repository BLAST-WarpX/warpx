#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse
from pathlib import Path

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from read_raw_data import _read_buffer
from scipy.constants import Boltzmann, c, elementary_charge, physical_constants
from scipy.integrate import quad
from scipy.optimize import brentq

parser = argparse.ArgumentParser()
parser.add_argument("--coupling", choices=["hybrid", "kinetic"], required=True)
parser.add_argument("--multigroup", action="store_true")
add_precision_arguments(parser)
args = parser.parse_args()
_, _, cross_dtype = precision_dtypes(args)


def load_plotfile(path: Path) -> dict[str, np.ndarray]:
    with open(path / "Header") as header:
        header.readline()
        n_fields = int(header.readline())
        field_names = [header.readline().strip() for _ in range(n_fields)]
    return _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), field_names)


initial_fields = load_plotfile(Path("diags/diag1000000"))
fields = load_plotfile(Path("diags/diag1000001"))

if args.multigroup:
    diffusion_groups = np.array(
        [
            np.sum(fields["radiation_diffusion_energy_g0"]),
            np.sum(fields["radiation_diffusion_energy_g1"]),
        ]
    )
else:
    diffusion_groups = np.array([np.sum(fields["radiation_diffusion_energy"])])
diffusion_energy = np.sum(diffusion_groups)
material_exchange = np.sum(fields["radiation_material_energy"])
realized_material = np.sum(
    fields["radiation_material_energy"] - initial_fields["radiation_material_energy"]
)
cross_solver_rtol = 3.0e-6 if cross_dtype == np.float32 else 3.0e-10
conservation_rtol = 1.0e-5 if cross_dtype == np.float32 else 2.0e-11

alpha_planck = 1.0e5
dt = 1.0e-10
relaxation = -np.expm1(-alpha_planck * c * dt)
radiation_constant = 4.0 * physical_constants["Stefan-Boltzmann constant"][0] / c

print(f"implicit diffusion energy: {diffusion_energy:.16e} J")
print(f"material exchange:         {material_exchange:.16e} J")
assert diffusion_energy > 0.0

if args.coupling == "hybrid":
    np.testing.assert_allclose(
        material_exchange, -diffusion_energy, rtol=conservation_rtol
    )
    initial_temperature = elementary_charge / Boltzmann
    electron_density = 1.0e20
    gamma = 5.0 / 3.0
    heat_capacity = electron_density * Boltzmann / (gamma - 1.0)
    initial_material_energy = heat_capacity * initial_temperature

    final_temperature = brentq(
        lambda temperature: (
            heat_capacity * temperature
            + relaxation * radiation_constant * temperature**4
            - initial_material_energy
        ),
        0.0,
        initial_temperature,
    )
    expected_diffusion = relaxation * radiation_constant * final_temperature**4
    measured_temperature = fields["Te"]
    measured_density = fields["rho"] / elementary_charge
    electron_energy_loss = np.sum(
        measured_density
        * Boltzmann
        * (initial_temperature - measured_temperature)
        / (gamma - 1.0)
        / measured_temperature.size
    )

    print(f"implicit final temperature: {final_temperature:.16e} K")
    np.testing.assert_allclose(
        diffusion_energy, expected_diffusion, rtol=cross_solver_rtol
    )
    np.testing.assert_allclose(
        measured_temperature, final_temperature, rtol=cross_solver_rtol
    )
    np.testing.assert_allclose(
        electron_energy_loss, diffusion_energy, rtol=conservation_rtol
    )

    planck_integral = np.pi**4 / 15.0
    boundary = elementary_charge / (Boltzmann * final_temperature)
    low_fraction = (
        quad(lambda x: x**3 / np.expm1(x), 0.0, boundary)[0] / planck_integral
    )
    expected_groups = expected_diffusion * np.array([low_fraction, 1.0 - low_fraction])
    np.testing.assert_allclose(
        diffusion_groups, expected_groups, rtol=cross_solver_rtol
    )
else:
    particle_energy = np.loadtxt("diags/particle_energy.txt")
    radiation_table = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
    assert radiation_table.shape[1] >= 17
    requested_material = -diffusion_energy
    residual = radiation_table[-1, -2]
    cumulative_residual = radiation_table[-1, -1]
    precision = cross_dtype
    field_roundoff = (
        32.0
        * np.finfo(precision).eps
        * max(
            abs(radiation_table[0, 2]),
            abs(radiation_table[-1, 2]),
            abs(realized_material),
            abs(radiation_table[-1, 5]),
            abs(radiation_table[-1, 6]),
        )
    )
    closure_residual = (
        radiation_table[0, 2] - radiation_table[-1, 2] - realized_material
    )
    np.testing.assert_allclose(
        residual, closure_residual, rtol=0.0, atol=max(1.0e-24, field_roundoff)
    )
    np.testing.assert_allclose(
        residual,
        requested_material - realized_material,
        rtol=0.0,
        atol=max(1.0e-24, field_roundoff),
    )
    np.testing.assert_allclose(
        cumulative_residual, residual, rtol=0.0, atol=max(1.0e-24, field_roundoff)
    )
    np.testing.assert_allclose(
        radiation_table[-1, 5],
        realized_material,
        rtol=0.0,
        atol=max(1.0e-24, field_roundoff),
    )
    np.testing.assert_allclose(
        radiation_table[-1, 6],
        realized_material,
        rtol=0.0,
        atol=max(1.0e-24, field_roundoff),
    )
    electron_energy_loss = particle_energy[0, 3] - particle_energy[-1, 3]
    cell_volume = 1.0 / fields["T_electrons"].size
    temperature = fields["T_electrons"] * elementary_charge / Boltzmann
    expected_cells = relaxation * radiation_constant * temperature**4 * cell_volume
    measured_cells = fields["radiation_diffusion_energy"].squeeze()

    # WarpX's kinetic temperature diagnostic uses the non-relativistic
    # momentum variance, whereas the conservative adapter rescales exact
    # relativistic particle energies.  At 10^4 K their difference is a few ppm.
    kinetic_temperature_rtol = 1.0e-5 if cross_dtype == np.float32 else 3.0e-6
    np.testing.assert_allclose(
        measured_cells, expected_cells, rtol=kinetic_temperature_rtol
    )
    np.testing.assert_allclose(
        electron_energy_loss,
        -realized_material,
        rtol=conservation_rtol,
        atol=max(1.0e-24, field_roundoff),
    )
    electron_radiation_drift = (
        particle_energy[-1, 3]
        - particle_energy[0, 3]
        + radiation_table[-1, 2]
        - radiation_table[0, 2]
    )
    particle_roundoff = (
        32.0
        * np.finfo(cross_dtype).eps
        * max(
            abs(particle_energy[0, 3]),
            abs(particle_energy[-1, 3]),
            abs(radiation_table[0, 2]),
            abs(radiation_table[-1, 2]),
        )
    )
    np.testing.assert_allclose(
        electron_radiation_drift,
        -residual,
        rtol=0.0,
        atol=max(1.0e-24, particle_roundoff),
    )
    print(f"kinetic numerical residual: {residual:.16e} J")
    assert particle_energy[-1, 3] > 0.0
