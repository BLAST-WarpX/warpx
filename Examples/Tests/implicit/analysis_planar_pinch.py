#!/usr/bin/env python3

# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from
# the script `inputs_test_1d_theta_implicit_planar_pinch`.
# and script `inputs_test_2d_theta_implicit_planar_pinch`.
# This simulates a planar pinch using the theta-implicit solver with
# the curl curl PC including the diagonal response from mass matrices.

import sys

import numpy as np
import yt
from scipy.constants import e, epsilon_0

newton_solver = np.loadtxt("diags/reduced_files/newton_solver.txt", skiprows=1)
num_steps = newton_solver[-1, 0]
total_newton_iters = newton_solver[-1, 3]
total_gmres_iters = newton_solver[-1, 7]

field_energy = np.loadtxt("diags/reduced_files/field_energy.txt", skiprows=1)
particle_energy = np.loadtxt("diags/reduced_files/particle_energy.txt", skiprows=1)
poynting_flux = np.loadtxt("diags/reduced_files/poynting_flux.txt", skiprows=1)

E_energy = field_energy[:, 3]
B_energy = field_energy[:, 4]
Efields = E_energy + B_energy
ele_energy = particle_energy[:, 3]
ion_energy = particle_energy[:, 4]
Eplasma = ele_energy + ion_energy

if poynting_flux.shape[1] == 10:
    print("2D simulation")
    gmres_iters_tol = 21
    Eout_lo_x = poynting_flux[:, 6]
    Eout_lo_z = poynting_flux[:, 7]
    Eout_hi_x = poynting_flux[:, 8]
    Eout_hi_z = poynting_flux[:, 9]
    dE_poynting = Eout_hi_x + Eout_lo_x + Eout_hi_z + Eout_lo_z
else:
    print("1D simulation")
    gmres_iters_tol = 10
    Eout_lo_x = poynting_flux[:, 4]
    Eout_hi_x = poynting_flux[:, 5]
    dE_poynting = Eout_hi_x + Eout_lo_x

# check that violation of energy conservation is below tolerance
dE = Efields + Eplasma + dE_poynting
rel_net_energy = np.abs(dE - dE[0]) / Eplasma
max_rel_net_energy = rel_net_energy.max()
rel_net_energy_tol = 1.0e-12
print(f"max relative delta energy : {max_rel_net_energy}")
print(f"relative delta energy tolerance : {rel_net_energy_tol}")
assert max_rel_net_energy < rel_net_energy_tol

# check that the number of gmres iterations is below tolerance
print(f"gmres iters per newton: {total_gmres_iters / total_newton_iters}")
print(f"gmres iters tolerance: {gmres_iters_tol}")
assert total_gmres_iters / total_newton_iters < gmres_iters_tol

# check that the number of newton iterations is below tolerance
newton_iters_tol = 5
print(f"newton iters per time step: {total_newton_iters / num_steps}")
print(f"newton iters tolerance: {newton_iters_tol}")
assert total_newton_iters / num_steps < newton_iters_tol

# check for machine precision conservation of charge density
n0 = 1.0e23

pltdir = sys.argv[1]
ds = yt.load(pltdir)
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

divE = data["boxlib", "divE"].value
rho = data["boxlib", "rho"].value

# compute local error in Gauss's law
drho = (rho - epsilon_0 * divE) / e / n0

# compute RMS error in charge conservation on the grid
# excluding the upper boundary where the insulator is located
drho_trimmed = drho[:-1, ...]
Ng = drho_trimmed.size
drho2_avg = (drho_trimmed**2).sum() / Ng
drho_rms = np.sqrt(drho2_avg)
tolerance_rel_charge = 1.0e-12
print(f"rms error in charge conservation: {drho_rms}")
print(f"tolerance: {tolerance_rel_charge}")
assert drho_rms < tolerance_rel_charge
