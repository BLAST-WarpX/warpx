#!/usr/bin/env python3

# Copyright 2024 Justin Angus, David Grote
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from the script `inputs_vandb_2d`.
# This simulates a 2D periodic plasma using the implicit solver
# with the Villasenor deposition using shape factor 2.

import numpy as np

field_energy = np.loadtxt("diags/reducedfiles/field_energy.txt", skiprows=1)
particle_energy = np.loadtxt("diags/reducedfiles/particle_energy.txt", skiprows=1)

total_energy = field_energy[:, 2] + particle_energy[:, 2]

delta_E = (total_energy - total_energy[0]) / total_energy[0]
max_delta_E = np.abs(delta_E).max()

# This case should have near machine precision conservation of energy
tolerance_rel_energy = 2.1e-14

print(f"max change in energy: {max_delta_E}")
print(f"tolerance: {tolerance_rel_energy}")

assert max_delta_E < tolerance_rel_energy
