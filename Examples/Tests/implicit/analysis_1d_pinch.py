#!/usr/bin/env python3

# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from
# the script `inputs_test_1d_planar_pinch_withPC`.
# This simulates a 1D planar pinch using the implicit solver with
# the curl curl PC including the diagonal response from mass matrices.

import numpy as np

newton_solver = np.loadtxt("diags/reduced_files/newton_solver.txt", skiprows=1)
gmres_iters = newton_solver[:, 5]

field_energy = np.loadtxt("diags/reduced_files/field_energy.txt", skiprows=1)
particle_energy = np.loadtxt("diags/reduced_files/particle_energy.txt", skiprows=1)
poynting_flux = np.loadtxt("diags/reduced_files/poynting_flux.txt", skiprows=1)

E_energy = field_energy[:, 3]
B_energy = field_energy[:, 4]
Efields = E_energy + B_energy
ele_energy = particle_energy[:, 3]
ion_energy = particle_energy[:, 4]
Eplasma = ele_energy + ion_energy

Eout_lo_x = poynting_flux[:, 4]
Eout_hi_x = poynting_flux[:, 5]

# check that violation of energy conservation is below tolerance
dE = Efields + Eplasma + Eout_hi_x + Eout_lo_x
rel_net_energy = np.abs(dE - dE[0]) / Eplasma
max_rel_net_energy = rel_net_energy.max()
rel_net_energy_tol = 1.0e-9
print(f"max relative delta energy : {max_rel_net_energy}")
print(f"relative delta energy tolerance : {rel_net_energy_tol}")
assert max_rel_net_energy < rel_net_energy_tol

# check that the maximum gmres iterations is below tolerance
max_gmres_iters_tol = 30
print(f"max gmres iters: {gmres_iters.max()}")
print(f"gmres iters tolerance: {max_gmres_iters_tol}")
assert gmres_iters.max() < max_gmres_iters_tol
