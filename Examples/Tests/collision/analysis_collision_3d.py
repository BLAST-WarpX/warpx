#!/usr/bin/env python3

# Copyright 2019-2020 Yinjian Zhao
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This script tests the collision module
# using electron-ion temperature relaxation in 3D.
# Initially, electrons and ions are both in equilibrium
# (gaussian) distributions, but have different temperatures.
# The electrons additionally have a drift speed in the x-direction
# equal to their thermal speed.
# Relaxation occurs to bring the velocity and temperature of each
# species to common values through collisions.
# The code was tested to be valid and an exponential fit with
# coefficients a and b to the temperature difference during the
# relaxation was obtained using a converged simulation with a
# 50X smaller time step than is used for this CI test.
# This automated test compares the results with the fit.
# Unrelated to the collision module, we also test the plotfile particle filter function in this
# analysis script.

# Possible running time: ~ 30.0 s

import glob
import sys

import numpy as np
import post_processing_utils
import yt
from scipy.constants import e, m_e

ng = 512
ne = ng * 200
ni = ng * 200
nptot = ne + ni

mi = m_e * 5.0

## In the first part of the test we verify that the output data is consistent with the exponential
## fit.

a_tolerance = 0.01
b_tolerance = 0.1

a_fit = 6.980918134261288
b_fit = -6.944721119341786e05

last_fn = sys.argv[1]
if last_fn[-1] == "/":
    last_fn = last_fn[:-1]
last_it = last_fn[-6:]  # i.e., 000100
prefix = last_fn[:-6]  # i.e., diags/diag1

# Collect all output files in fn_list (names match pattern prefix + arbitrary number)
fn_list = glob.glob(prefix + "*[0-9]")
fn_list.sort(key=lambda fn: int(fn[-6:]))

time_arr = []
Te_arr = []
Ti_arr = []

for fn in fn_list:
    # load file
    ds = yt.load(fn)
    time = ds.current_time.to_value()
    ad = ds.all_data()
    pxe = ad["electron", "particle_momentum_x"].to_ndarray()
    pye = ad["electron", "particle_momentum_y"].to_ndarray()
    pze = ad["electron", "particle_momentum_z"].to_ndarray()
    pxi = ad["ion", "particle_momentum_x"].to_ndarray()
    pyi = ad["ion", "particle_momentum_y"].to_ndarray()
    pzi = ad["ion", "particle_momentum_z"].to_ndarray()

    # compute mean drift velocity
    vxe = np.mean(pxe) / m_e
    vye = np.mean(pye) / m_e
    vze = np.mean(pze) / m_e
    vxi = np.mean(pxi) / mi
    vyi = np.mean(pyi) / mi
    vzi = np.mean(pzi) / mi

    # compute thermal velocity
    vtesq = np.mean((pxe / m_e - vxe) ** 2)
    vtesq += np.mean((pye / m_e - vye) ** 2)
    vtesq += np.mean((pze / m_e - vze) ** 2)
    vtisq = np.mean((pxi / mi - vxi) ** 2)
    vtisq += np.mean((pyi / mi - vyi) ** 2)
    vtisq += np.mean((pzi / mi - vzi) ** 2)

    Te_eV = m_e / e * vtesq / 3.0
    Ti_eV = mi / e * vtisq / 3.0

    time_arr.append(time)
    Te_arr.append(Te_eV)
    Ti_arr.append(Ti_eV)

time_arr = np.array(time_arr)
Te_arr = np.array(Te_arr)
Ti_arr = np.array(Ti_arr)

# Compute fit in time window where temerature difference
# displays exponential decay
deltaT = Te_arr - Ti_arr
mask = (time_arr > 1.0e-6) & (time_arr < 4.0e-6)
logTdiff = np.log(deltaT[mask])
tfit = time_arr[mask]
b, a = np.polyfit(tfit, logTdiff, 1)

a_error = np.abs((a - a_fit) / a_fit)
b_error = np.abs((b - b_fit) / b_fit)

print("a_error = ", a_error)
print("a_tolerance = ", a_tolerance)
print("b_error = ", b_error)
print("b_tolerance = ", b_tolerance)
assert a_error < a_tolerance
assert b_error < b_tolerance

## In the second part of the test, we verify that the diagnostic particle filter function works as
## expected. For this, we only use the last simulation timestep.

dim = "3d"
species_name = "electron"

parser_filter_fn = "diags/diag_parser_filter" + last_it
parser_filter_expression = "(px*py*pz < 0) * (nptot.sqrt(x**2+y**2+z**2)<100)"
post_processing_utils.check_particle_filter(
    last_fn, parser_filter_fn, parser_filter_expression, dim, species_name
)

uniform_filter_fn = "diags/diag_uniform_filter" + last_it
uniform_filter_expression = "ids%11 == 0"
post_processing_utils.check_particle_filter(
    last_fn, uniform_filter_fn, uniform_filter_expression, dim, species_name
)

random_filter_fn = "diags/diag_random_filter" + last_it
random_fraction = 0.88
post_processing_utils.check_random_filter(
    last_fn, random_filter_fn, random_fraction, dim, species_name
)
