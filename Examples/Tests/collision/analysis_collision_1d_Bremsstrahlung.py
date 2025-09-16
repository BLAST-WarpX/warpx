#!/usr/bin/env python3

# Copyright 2025 David Grote
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from the script `inputs_test_1d_collision_z_Bremsstrahlung`.
# run locally: python analysis_collision_1d_Bremsstrahlung.py diags/diag1000600/
#
# This is a 1D Bremsstrahlung collision of electrons on Boron ions
# producing photons. This checks that the appropriate number of
# photons are created with the correct distribution.
#
import os
import sys

import numpy as np
from scipy import constants

# this will be the name of the plot file
last_fn = sys.argv[1]

particle_energy = np.loadtxt(
    os.path.join("diags", "reducedfiles", "particle_energy.txt"), skiprows=1
)
particle_momentum = np.loadtxt(
    os.path.join("diags", "reducedfiles", "particle_momentum.txt"), skiprows=1
)
particle_number = np.loadtxt(
    os.path.join("diags", "reducedfiles", "particle_number.txt"), skiprows=1
)

total_energy = particle_energy[:, 2]
electron_energy = particle_energy[:, 3]
ion_energy = particle_energy[:, 4]
photon_energy = particle_energy[:, 5]

total_momentum = particle_momentum[:, 2]
electron_momentum = particle_momentum[:, 3]
ion_momentum = particle_momentum[:, 4]
photon_momentum = particle_momentum[:, 5]

energy_tolerance = 1.0e-11

print(f"initial total energy = {total_energy[0]}")
print(f"final total energy   = {total_energy[-1]}")
print(f"energy tolerance = {energy_tolerance}")

dE_total = np.abs(total_energy[-1] - total_energy[0]) / total_energy[0]
print(f"change in total energy = {dE_total}")
assert dE_total < energy_tolerance

print()

momentum_tolerance = 1.0e-12

print(f"initial total momentum = {total_momentum[0]}")
print(f"final total momentum   = {total_momentum[-1]}")
print(f"momentum tolerance = {momentum_tolerance}")

dP_total = np.abs(total_momentum[-1] - total_momentum[0]) / total_momentum[0]
print(f"change in total momentum = {dP_total}")
assert dP_total < momentum_tolerance

print()

dt = 1.0e-2 * 1.0e-15
Z = 5
n_i = 5.47e31
n_e = 5.47e30
L = 1.0e-6  # 1 micron
T1 = 1.0e6  # 1 MeV
N_e = n_e * L  # number of electrons
m_e_eV = constants.m_e * constants.c**2 / constants.e
gamma = T1 / m_e_eV + 1.0
gamma_beta = np.sqrt(gamma**2 - 1.0)
beta = gamma_beta / gamma

phirad = 6.761  # from Seltzer and Berger for 1 MeV electron and Boron


dEdx_simulation = (
    (particle_energy[0, 3] - particle_energy[1:, 3])
    / particle_energy[1:, 0]
    / (beta * constants.c * dt)
    / constants.e
    / N_e
)

Boron_weight = 20065.0 * constants.m_e
r_e = (
    1.0
    / (4.0 * constants.pi * constants.epsilon_0)
    * (constants.e**2 / (constants.m_e * constants.c**2))
)
dEdx = n_i * constants.alpha * r_e**2 * Z**2 * (T1 + m_e_eV) * phirad

dEdx_difference = np.abs(dEdx_simulation[-1] / dEdx - 1.0)
dEdx_tolerance = 0.03

print(f"dE/dx analytic  = {dEdx}")
print(f"dE/dx simulated = {dEdx_simulation[-1]}")
print(f"dE/dx simulated/analytic = {dEdx_simulation[-1] / dEdx}")
print(f"dE/dx tolerance = {dEdx_tolerance}")
print(f"dE/dx difference = {dEdx_difference}")
assert dEdx_difference < dEdx_tolerance

sigma_total = 1.818e-28  # Calculated from table with k cutoff=1.e-4
N_photon = n_e * n_i * L * beta * constants.c * sigma_total * dt
new_photons_tolerance = 0.02
new_photons_difference = np.abs(
    particle_number[-1, -1] / (particle_energy[-1, 0] * N_photon) - 1.0
)
print(f"New photons per step simulated = {particle_number[-1, -1]}")
print(f"New photons per step analytic  = {particle_energy[-1, 0] * N_photon}")
print(
    f"New photons per step simulated/analytic = {particle_number[-1, -1] / (particle_energy[-1, 0] * N_photon)}"
)
print(f"New photons per step toleraance = {new_photons_tolerance}")
print(f"New photons per step difference = {new_photons_difference}")
assert new_photons_difference < new_photons_tolerance
