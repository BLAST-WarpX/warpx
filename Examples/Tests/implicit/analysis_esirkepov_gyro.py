#!/usr/bin/env python3

# Copyright 2026
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# Regression test for zero-displacement implicit Esirkepov field gathering.
# Two counter-streaming uniform electron populations gyrate in a uniform
# external grid magnetic field B_z with zero axial velocity (z_new == z_old).
# With the zero-displacement divided-difference bug, higher-order shapes
# multiply the gathered B_z by the particle shape order (2x for quadratic).
import math

import numpy as np
from scipy.constants import c

mom = np.loadtxt("diags/reduced_files/particle_momentum.txt", skiprows=1)
field_energy = np.loadtxt("diags/reduced_files/field_energy.txt", skiprows=1)
particle_energy = np.loadtxt("diags/reduced_files/particle_energy.txt", skiprows=1)

# Columns 5 and 6 are px and py of the first species ("plus")
px = mom[:, 5]
py = mom[:, 6]
time = mom[:, 1]

phi = 0.0
for i in range(len(time) - 1):
    phi += math.atan2(
        px[i] * py[i + 1] - py[i] * px[i + 1],
        px[i] * px[i + 1] + py[i] * py[i + 1],
    )

# Analytical relativistic gyrofrequency in units of omega_p = c / (1 m)
u0 = 0.375
b_norm = 0.75
omega_continuum = b_norm / math.sqrt(1.0 + u0 * u0)
dt_norm = (time[-1] - time[0]) * c / (len(time) - 1)
omega_discrete = 2.0 * math.atan(0.5 * omega_continuum * dt_norm) / dt_norm
omega_measured = phi / ((time[-1] - time[0]) * c)

rel_err_continuum = abs(omega_measured - omega_continuum) / omega_continuum
rel_err_discrete = abs(omega_measured - omega_discrete) / omega_discrete

total_energy = field_energy[:, 2] + particle_energy[:, 2]
max_rel_energy_drift = np.abs(total_energy - total_energy[0]).max() / total_energy[0]

print(f"Continuum gyrofrequency (omega_p): {omega_continuum:.12g}")
print(f"Discrete  gyrofrequency (omega_p): {omega_discrete:.12g}")
print(f"Measured  gyrofrequency (omega_p): {omega_measured:.12g}")
print(f"Relative error vs continuum:       {rel_err_continuum:.6e}")
print(f"Relative error vs discrete:        {rel_err_discrete:.6e}")
print(f"Max relative total energy drift:   {max_rel_energy_drift:.6e}")

assert rel_err_continuum < 1.0e-3, (
    f"Measured gyrofrequency {omega_measured} deviates from continuum {omega_continuum}"
)
assert rel_err_discrete < 1.0e-4, (
    f"Measured gyrofrequency {omega_measured} deviates from discrete {omega_discrete}"
)
assert max_rel_energy_drift < 1.0e-5, (
    f"Total energy drift {max_rel_energy_drift} exceeds tolerance"
)
