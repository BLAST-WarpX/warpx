#!/usr/bin/env python3

# Copyright 2024 Justin Angus
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This script analyses conservation of energy and charge from simulations of
# a uniform thermal plasma using the exactly energy-conserving EM implicit method
# in 1D, 2D, or 3D with any combination of periodic and symmetry boundaries.
import sys

import numpy as np
import yt
from scipy.constants import e, epsilon_0

field_energy = np.loadtxt("diags/reduced_files/field_energy.txt", skiprows=1)
particle_energy = np.loadtxt("diags/reduced_files/particle_energy.txt", skiprows=1)

total_energy = field_energy[:, 2] + particle_energy[:, 2]

delta_E = (total_energy - total_energy[0]) / total_energy[0]
max_delta_E = np.abs(delta_E).max()

# This case should have near machine precision conservation of charge and energy
tolerance_rel_energy = 2.0e-14
tolerance_rel_charge = 2.0e-14

print(f"max change in energy: {max_delta_E}")
print(f"tolerance: {tolerance_rel_energy}")

assert max_delta_E < tolerance_rel_energy

# check for machine precision conservation of charge density

pltdir = sys.argv[1]
ds = yt.load(pltdir)
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

divE = data["boxlib", "divE"].value
rho = data["boxlib", "rho"].value
num = data["boxlib", "num_electrons"].value
Lx = ds.domain_right_edge[0] - ds.domain_left_edge[0]
Ly = ds.domain_right_edge[1] - ds.domain_left_edge[1]
Lz = ds.domain_right_edge[2] - ds.domain_left_edge[2]
dV = Lx.value * Ly.value * Lz.value
ne0 = num.sum() / dV

# compute local error in Gauss's law
drho = (rho - epsilon_0 * divE) / e / ne0

# compute RMS on in error on the grid
nX = drho.shape[0]
nY = drho.shape[1]
nZ = drho.shape[2]
drho2_avg = (drho**2).sum() / (nX * nY * nZ)
drho_rms = np.sqrt(drho2_avg)

print(f"rms error in charge conservation: {drho_rms}")
print(f"tolerance: {tolerance_rel_charge}")

assert drho_rms < tolerance_rel_charge
