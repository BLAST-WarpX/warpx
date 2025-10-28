#!/usr/bin/env python3

# Copyright 2018-2019 Andrew Myers, Jean-Luc Vay, Maxence Thevenet
# Remi Lehe
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


import sys

import numpy as np
import scipy.constants as scc
import yt

yt.funcs.mylog.setLevel(0)

filename = sys.argv[1]

############################
### INITIAL LASER ENERGY ###
############################
energy_start = 9.1301289517e-08

##########################
### FINAL LASER ENERGY ###
##########################
ds = yt.load(filename)
all_data_level_0 = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
Bx = all_data_level_0["boxlib", "Bx"].v.squeeze()
By = all_data_level_0["boxlib", "By"].v.squeeze()
Bz = all_data_level_0["boxlib", "Bz"].v.squeeze()
Ex = all_data_level_0["boxlib", "Ex"].v.squeeze()
Ey = all_data_level_0["boxlib", "Ey"].v.squeeze()
Ez = all_data_level_0["boxlib", "Ez"].v.squeeze()
energyE = np.sum(scc.epsilon_0 / 2 * (Ex**2 + Ey**2 + Ez**2))
energyB = np.sum(1.0 / scc.mu_0 / 2 * (Bx**2 + By**2 + Bz**2))
energy_end = energyE + energyB

Reflectivity = energy_end / energy_start
Reflectivity_theory = 1.8015e-06

print("Reflectivity: %s" % Reflectivity)
print("Reflectivity_theory: %s" % Reflectivity_theory)

error_rel = abs(Reflectivity - Reflectivity_theory) / Reflectivity_theory
tolerance_rel = 5.0 / 100

print("error_rel    : " + str(error_rel))
print("tolerance_rel: " + str(tolerance_rel))

assert error_rel < tolerance_rel
