#!/usr/bin/env python3

# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from
# the script `inputs_test_3d_collision_pulsed_decay`.
# A neutral gas decays into electron-ion plasma using the pulsed_decay
# collision module with a guassian decay rate. This script checks
# that the final weight of the product ions agrees with a 0D model.

import sys

import numpy as np

# 0D model
Amp = 5.975e7
n10 = 2.2e17
tau = 20.0e-9
shift = 150.0e-9
dt = 0.001e-9

time_arr = np.arange(0, 250e-9 + dt, dt)
decay_rate = Amp * np.exp(-(((time_arr - shift) / tau) ** 2) / 2)
n1_arr = np.zeros_like(time_arr)
n1_arr[0] = n10

for it in range(1, len(time_arr)):
    n1_arr[it] = n1_arr[it - 1] * np.exp(-decay_rate[it] * dt)

nI_arr = n10 - n1_arr

fn = sys.argv[1]
particle_number_diag = np.loadtxt(fn + "/reduced_files/particle_number.txt", skiprows=1)

step = particle_number_diag[:, 0]
time = particle_number_diag[:, 1]
ele_number = particle_number_diag[:, 3]
ion_number = particle_number_diag[:, 4]
neu_number = particle_number_diag[:, 5]
ele_weight = particle_number_diag[:, 7]
ion_weight = particle_number_diag[:, 8]
neu_weight = particle_number_diag[:, 9]

# check that the total weight of the product ions matches the 0D model
ion_weight_end = ion_weight[-1]
ion_weight_rtol = 0.5  # half a percent
ion_weight_error = 100 * np.abs(nI_arr[-1] - ion_weight_end) / nI_arr[-1]
print(f"ion weight (WarpX) = {ion_weight_end}")
print(f"ion weight (0D)    = {nI_arr[-1]}")
print(f"weight error       = {ion_weight_error}")
assert ion_weight_eror < ion_weight_rtol
