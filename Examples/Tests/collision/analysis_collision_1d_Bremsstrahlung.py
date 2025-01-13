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
# This is a 1D Bremsstrahlung collision of electrons on Borons ions
# producing photons. This checks that the appropriate number of
# photons are created with the correct distribution.
#
import sys
import os

import numpy as np
from scipy import constants

# this will be the name of the plot file
last_fn = sys.argv[1]

particle_energy = np.loadtxt(os.path.join("diags", "reducedfiles", "particle_energy.txt"), skiprows=1)
particle_number = np.loadtxt(os.path.join("diags", "reducedfiles", "particle_number.txt"), skiprows=1)
photon_energy = np.loadtxt(os.path.join("diags", "reducedfiles", "photon_energy.txt"), skiprows=1)

total_energy = particle_energy[:,2]
electron_energy = particle_energy[:,3]
ion_energy = particle_energy[:,4]
photon_energy = particle_energy[:,5]

energy_tolerance = 1.e-12

print(f"initial total energy = {total_energy[0]}")
print(f"final total energy = {total_energy[-1]}")

dE_total = np.abs(total_energy[-1] - total_energy[0])/total_energy[0]
print(f"change in total energy = {dE_total}")
assert dE_total < energy_tolerance

print(f"initial electron energy = {electron_energy[0]}")
print(f"final electron energy = {electron_energy[-1]}")
print(f"final photon energy = {photon_energy[-1]}")

dE_electron_photon = np.abs(electron_energy[-1] + photon_energy[-1] - electron_energy[0])/electron_energy[0]
print(f"electron, photon energy change = {dE_electron_photon}")
assert dE_electron_photon < energy_tolerance


print()

dt = 1.e-2*1.e-15
Z = 5
n_i = 5.47e31
n_e = 5.47e30
L = 1.e-6  # 1 micron
T1 = 1.e6  # 1 MeV
N_e = n_e*L  # number of electrons
m_e_eV = constants.m_e*constants.c**2/constants.e
gamma = T1/m_e_eV + 1.
gamma_beta = np.sqrt(gamma**2 - 1.)
beta = gamma_beta/gamma

phirad = 6.761  # from Seltzer and Berger for 1 MeV electron and Boron


dEdx_simulation = (particle_energy[0,3] - particle_energy[1:,3])/particle_energy[1:,0]/(beta*constants.c*dt)/constants.e/N_e

Boron_weight = 20065.0*constants.m_e
r_e = 1./(4.*constants.pi*constants.epsilon_0)*(constants.e**2/(constants.m_e*constants.c**2))
dEdx = n_i*constants.alpha*r_e**2*Z**2*(T1 + m_e_eV)*phirad

print(f"dE/dx analytic = {dEdx}")
print(F"dE/dx simulated = {dEdx_simulation[-1]}")
print(f"dE/dx simulated/analytic = {dEdx_simulation[-1]/dEdx}")
assert np.abs(dEdx_simulation[-1]/dEdx - 1.) < 0.03

sigam_total = 1.698e-28  # Calculated from table with k cutoff=1.e-4
N_photon = n_e*n_i*L*beta*constants.c*sigam_total*dt
print(f"New photons per step simulated/analytic = {particle_number[-1,-1]/(particle_energy[-1,0]*N_photon)}")
assert np.abs(particle_number[-1,-1]/(particle_energy[-1,0]*N_photon) - 1.) < 0.03


