#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from read_raw_data import _read_buffer
from scipy.constants import Boltzmann, c, elementary_charge, physical_constants
from scipy.integrate import quad

parser = argparse.ArgumentParser()
add_precision_arguments(parser)
args = parser.parse_args()
_, _, cross_dtype = precision_dtypes(args)

data = np.loadtxt("diags/radiation_energy.txt")
final = data[-1]

temperature = elementary_charge / Boltzmann
radiation_constant = 4.0 * physical_constants["Stefan-Boltzmann constant"][0] / c
equilibrium_energy = radiation_constant * temperature**4
dt = 1.0e-10

planck_integral = np.pi**4 / 15.0
low_fraction = quad(lambda x: x**3 / np.expm1(x), 0.0, 1.0)[0] / planck_integral
fractions = np.array([low_fraction, 1.0 - low_fraction])
opacities = np.array([10.0, 30.0])
expected = equilibrium_energy * fractions * (1.0 - np.exp(-opacities * c * dt))

diffusion_group_energy = final[9:11]
diffusion_energy = final[4]
material_exchange = final[5]
streaming_energy_gain = final[3] - data[0, 3]
measured_group_energy = np.array([diffusion_group_energy[0], streaming_energy_gain])

plotfile = "diags/diag1000001"
with open(f"{plotfile}/Header") as header:
    header.readline()
    n_fields = int(header.readline())
    field_names = [header.readline().strip() for _ in range(n_fields)]
fields = _read_buffer(plotfile, f"{plotfile}/Level_0/Cell_H", field_names)
cross_physics_rtol = 8.0e-6 if cross_dtype == np.float32 else 4.0e-13
cross_sum_rtol = 2.0e-6 if cross_dtype == np.float32 else 2.0e-15
grid_group_energy = np.array(
    [
        np.sum(fields["radiation_diffusion_energy_g0"]),
        np.sum(fields["radiation_diffusion_energy_g1"]),
    ]
)

print(f"expected group energies: {expected}")
print(f"measured group energies: {measured_group_energy}")
print(f"diffusion group energies: {diffusion_group_energy}")
print(f"grid group energies:      {grid_group_energy}")
print(f"material exchange:       {material_exchange:.16e} J")

np.testing.assert_allclose(
    measured_group_energy[0], expected[0], rtol=cross_physics_rtol
)
np.testing.assert_allclose(
    measured_group_energy[1], expected[1], rtol=cross_physics_rtol
)
np.testing.assert_allclose(
    diffusion_energy, diffusion_group_energy[0], rtol=cross_sum_rtol
)
np.testing.assert_allclose(diffusion_group_energy[1], 0.0, atol=1.0e-30)
np.testing.assert_allclose(
    grid_group_energy, diffusion_group_energy, rtol=cross_sum_rtol
)
np.testing.assert_allclose(
    material_exchange, -np.sum(measured_group_energy), rtol=cross_physics_rtol
)
assert streaming_energy_gain > diffusion_energy > 0.0
