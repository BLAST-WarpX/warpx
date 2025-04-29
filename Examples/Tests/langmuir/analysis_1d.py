#!/usr/bin/env python3

# Copyright 2019-2022 Jean-Luc Vay, Maxence Thevenet, Remi Lehe, Prabhat Kumar, Axel Huebl
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from
# the script `inputs.multi.rt`. This simulates a 1D periodic plasma wave.
# The electric field in the simulation is given (in theory) by:
# $$ E_z = \epsilon \,\frac{m_e c^2 k_z}{q_e}\sin(k_z z)\sin( \omega_p t)$$
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt

yt.funcs.mylog.setLevel(50)

import numpy as np
from scipy.constants import c, e, epsilon_0, m_e

# test name
test_name = os.path.split(os.getcwd())[1]

# this will be the name of the plot file
fn = sys.argv[1]

# Parse test name and check if current correction (psatd.current_correction=1) is applied
current_correction = True if re.search("current_correction", test_name) else False

# Parse test name and check if Vay current deposition (algo.current_deposition=vay) is used
vay_deposition = True if re.search("vay_deposition", test_name) else False

# Parameters (these parameters must match the parameters in `inputs.multi.rt`)
epsilon = 0.01
n = 4.0e24
n_osc_z = 2
zmin = -20e-6
zmax = 20.0e-6
Nz = 128

# Wave vector of the wave
kz = 2.0 * np.pi * n_osc_z / (zmax - zmin)
# Plasma frequency
wp = np.sqrt((n * e**2) / (m_e * epsilon_0))

k = {"Ez": kz}
cos = {"Ez": (1, 1, 0)}


def get_contribution(is_cos, k):
    du = (zmax - zmin) / Nz
    u = zmin + du * (0.5 + np.arange(Nz))
    if is_cos == 1:
        return np.cos(k * u)
    else:
        return np.sin(k * u)


def get_theoretical_field(field, t):
    amplitude = epsilon * (m_e * c**2 * k[field]) / e * np.sin(wp * t)
    cos_flag = cos[field]
    z_contribution = get_contribution(cos_flag[2], kz)

    E = amplitude * z_contribution

    return E


# Read the file
ds = yt.load(fn)
t0 = ds.current_time.to_value()
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
# Check the validity of the fields
error_rel = 0
for field in ["Ez"]:
    E_sim = data[("mesh", field)].to_ndarray()[:, 0, 0]
    E_th = get_theoretical_field(field, t0)
    max_error = abs(E_sim - E_th).max() / abs(E_th).max()
    print("%s: Max error: %.2e" % (field, max_error))
    error_rel = max(error_rel, max_error)

# Plot the last field from the loop (Ez at iteration 80)
plt.subplot2grid((1, 2), (0, 0))
plt.plot(E_sim)
# plt.colorbar()
plt.title("Ez, last iteration\n(simulation)")
plt.subplot2grid((1, 2), (0, 1))
plt.plot(E_th)
# plt.colorbar()
plt.title("Ez, last iteration\n(theory)")
plt.tight_layout()
plt.savefig("langmuir_multi_1d_analysis.png")

tolerance_rel = 0.05

print("error_rel    : " + str(error_rel))
print("tolerance_rel: " + str(tolerance_rel))

assert error_rel < tolerance_rel

# Check relative L-infinity spatial norm of rho/epsilon_0 - div(E) when
# current correction (psatd.do_current_correction=1) is applied or when
# Vay current deposition (algo.current_deposition=vay) is used
if current_correction or vay_deposition:
    rho = data[("boxlib", "rho")].to_ndarray()
    divE = data[("boxlib", "divE")].to_ndarray()
    error_rel = np.amax(np.abs(divE - rho / epsilon_0)) / np.amax(
        np.abs(rho / epsilon_0)
    )
    tolerance = 1.0e-9
    print("Check charge conservation:")
    print("error_rel = {}".format(error_rel))
    print("tolerance = {}".format(tolerance))
    assert error_rel < tolerance
