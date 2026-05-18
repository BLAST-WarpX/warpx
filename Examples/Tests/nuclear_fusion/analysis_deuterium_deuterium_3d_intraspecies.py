#!/usr/bin/env python3

# Copyright 2019-2023
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# Initialize a deuterium-deuterium thermal plasma similar to that in
# section 4.1 of [1], with fusion multiplication factor 1e10,
# 1e4 particles per cell, density 1e20/cm^3, temperature 20 keV.
# Check that the simulation reactivity matches the analytical value
# reported in Table VIII of [2], namely 2.603e-18 cm^3/s.
#
# Reference:
# [1] Higginson, D.P., Link, A. and Schmidt, A., 2019.
#     A pairwise nuclear fusion algorithm for weighted particle-in-cell
#     plasma simulations. Journal of Computational Physics, 388, pp.439-453.
#     DOI: https://doi.org/10.1016/j.jcp.2019.03.020
#
# [2] Bosch, H.S. and Hale, G.M., 1992.
#     Improved formulas for fusion cross-sections and thermal reactivities.
#     Nuclear fusion, 32(4), p.611.
#     DOI: https://doi.org/10.1088/0029-5515/32/4/I07

import numpy as np

# Load data from reduced diagnostics (physical time and neutron weights)
time = np.loadtxt("./reduced_diags/particle_number.txt", usecols=1)
neutron = np.loadtxt("./reduced_diags/particle_number.txt", usecols=9)

# Compute reactivity in units of cm^3/s as in equation (61) of [1]
dY_dt = np.abs(neutron[-1] - neutron[0]) / (time[-1] - time[0])
delta_ij = 1  # reactants of the same species
nD = 1e26  # density in 1/m^3
V = (2e-3) ** 3  # simulation volume in m^3
sigma = dY_dt * (1 + delta_ij) / (nD**2) / V * (1e2) ** 3

sigma_th = 2.603e-18
error = np.abs(sigma - sigma_th) / sigma_th
tolerance = 2e-2
print("error = ", error)
print("tolerance = ", tolerance)
assert error < tolerance
