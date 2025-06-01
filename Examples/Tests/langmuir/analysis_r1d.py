#!/usr/bin/env python3

# Copyright 2019 David Grote, Maxence Thevenet
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from
# the script `inputs_test_rcylinder_langmuir_multi` and `inputs_test_rspherical_langmuir_multi`.
# This simulates a 1d plasma wave using either cylindrical or spherical coordinates.
# The electric field in the simulation is given (in theory) by:
# $$ E_r = -\partial_r \phi = \epsilon \,\frac{mc^2}{e}\frac{2\,r}{w_0^2} \exp\left(-\frac{r^2}{w_0^2}\right) \sin(\omega_p t)
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

# this will be the name of the plot file
fn = sys.argv[1]

test_name = os.path.split(os.getcwd())[1]

# Parse test name and check if current correction (psatd.current_correction) is applied
current_correction = True if re.search("current_correction", fn) else False

# Parameters (these parameters must match the parameters in `inputs_rcylinder`)
epsilon = 0.01
n = 2.0e24
w0 = 5.0e-6
rmin = 0e-6
rmax = 20.0e-6
Nr = 64

# Plasma frequency
wp = np.sqrt((n * e**2) / (m_e * epsilon_0))
kp = wp / c


def Er(r, epsilon, w0, wp, t):
    """
    Return the radial electric field as an array
    of the same length as r, in the half-plane theta=0
    """
    Er_array = (
        epsilon
        * m_e
        * c**2
        / e
        * 2
        * r
        / w0**2
        * np.exp(-(r**2) / w0**2)
        * np.sin(wp * t)
    )
    return Er_array


# Read the file
ds = yt.load(fn)
t0 = ds.current_time.to_value()
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

# Get cell centered coordinates
dr = (rmax - rmin) / Nr
coords = np.indices([Nr], "d")
rr = rmin + (coords[0] + 0.5) * dr

# Check the validity of the fields
overall_max_error = 0
Er_sim = data[("boxlib", "Er")].to_ndarray()[:, 0, 0]
Er_th = Er(rr, epsilon, w0, wp, t0)
max_error = abs(Er_sim - Er_th).max() / abs(Er_th).max()
print("Er: Max error: %.2e" % (max_error))
overall_max_error = max(overall_max_error, max_error)

# Plot the last field from the loop (Er at iteration 40)
plt.subplot2grid((1, 2), (0, 0))
plt.plot(rr, Er_sim)
plt.title("Er, last iteration\n(simulation)")
plt.subplot2grid((1, 2), (0, 1))
plt.plot(rr, Er_th)
plt.title("Er, last iteration\n(theory)")
plt.tight_layout()
plt.savefig(test_name + "_analysis.png")

error_rel = overall_max_error

tolerance_rel = 0.12

print("error_rel    : " + str(error_rel))
print("tolerance_rel: " + str(tolerance_rel))

assert error_rel < tolerance_rel
