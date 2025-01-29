#!/usr/bin/env python3

# Copyright 2019 David Grote, Maxence Thevenet
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from
# the script `inputs.multi.rz.rt`. This simulates a RZ periodic plasma wave.
# The electric field in the simulation is given (in theory) by:
# $$ E_r = -\partial_r \phi = \epsilon \,\frac{mc^2}{e}\frac{2\,r}{w_0^2} \exp\left(-\frac{r^2}{w_0^2}\right) \sin(k_0 z) \sin(\omega_p t)
# $$ E_z = -\partial_z \phi = - \epsilon \,\frac{mc^2}{e} k_0 \exp\left(-\frac{r^2}{w_0^2}\right) \cos(k_0 z) \sin(\omega_p t)
# Unrelated to the Langmuir waves, we also test the plotfile particle filter function in this
# analysis script.
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt

yt.funcs.mylog.setLevel(50)

import numpy as np
import post_processing_utils
from scipy.constants import c, e, epsilon_0, m_e

# this will be the name of the plot file
fn = sys.argv[1]

# test name
test_name = os.path.split(os.getcwd())[1]

# Parse test name and check if current correction (psatd.current_correction) is applied
current_correction = True if re.search("current_correction", test_name) else False

# Parameters (these parameters must match the parameters in `inputs.multi.rz.rt`)
epsilon = 0.01
n = 2.0e24
w0 = 5.0e-6
n_osc_z = 2
rmin = 0e-6
rmax = 20.0e-6
Nr = 64
zmin = -20e-6
zmax = 20.0e-6
Nz = 128

# Wave vector of the wave
k0 = 2.0 * np.pi * n_osc_z / (zmax - zmin)
# Plasma frequency
wp = np.sqrt((n * e**2) / (m_e * epsilon_0))
kp = wp / c


def Er(z, r, epsilon, k0, w0, wp, t):
    """
    Return the radial electric field as an array
    of the same length as z and r, in the half-plane theta=0
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
        * np.sin(k0 * z)
        * np.sin(wp * t)
    )
    return Er_array


def Ez(z, r, epsilon, k0, w0, wp, t):
    """
    Return the longitudinal electric field as an array
    of the same length as z and r, in the half-plane theta=0
    """
    Ez_array = (
        -epsilon
        * m_e
        * c**2
        / e
        * k0
        * np.exp(-(r**2) / w0**2)
        * np.cos(k0 * z)
        * np.sin(wp * t)
    )
    return Ez_array


# Read the file
ds = yt.load(fn)
t0 = ds.current_time.to_value()
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

# Get cell centered coordinates
dr = (rmax - rmin) / Nr
dz = (zmax - zmin) / Nz
coords = np.indices([Nr, Nz], "d")
rr = rmin + (coords[0] + 0.5) * dr
zz = zmin + (coords[1] + 0.5) * dz

# Check the validity of the fields
overall_max_error = 0
Er_sim = data[("boxlib", "Er")].to_ndarray()[:, :, 0]
Er_th = Er(zz, rr, epsilon, k0, w0, wp, t0)
max_error = abs(Er_sim - Er_th).max() / abs(Er_th).max()
print("Er: Max error: %.2e" % (max_error))
overall_max_error = max(overall_max_error, max_error)

Ez_sim = data[("boxlib", "Ez")].to_ndarray()[:, :, 0]
Ez_th = Ez(zz, rr, epsilon, k0, w0, wp, t0)
max_error = abs(Ez_sim - Ez_th).max() / abs(Ez_th).max()
print("Ez: Max error: %.2e" % (max_error))
overall_max_error = max(overall_max_error, max_error)

# Plot the last field from the loop (Ez at iteration 40)
plt.subplot2grid((1, 2), (0, 0))
plt.imshow(Ez_sim)
plt.colorbar()
plt.title("Ez, last iteration\n(simulation)")
plt.subplot2grid((1, 2), (0, 1))
plt.imshow(Ez_th)
plt.colorbar()
plt.title("Ez, last iteration\n(theory)")
plt.tight_layout()
plt.savefig(test_name + "_analysis.png")

error_rel = overall_max_error

tolerance_rel = 0.12

print("error_rel    : " + str(error_rel))
print("tolerance_rel: " + str(tolerance_rel))

assert error_rel < tolerance_rel

# Check charge conservation (relative L-infinity norm of error) with current correction
if current_correction:
    divE = data[("boxlib", "divE")].to_ndarray()
    rho = data[("boxlib", "rho")].to_ndarray() / epsilon_0
    error_rel = np.amax(np.abs(divE - rho)) / max(np.amax(divE), np.amax(rho))
    tolerance = 1.0e-9
    print("Check charge conservation:")
    print("error_rel = {}".format(error_rel))
    print("tolerance = {}".format(tolerance))
    assert error_rel < tolerance


## In the final past of the test, we verify that the diagnostic particle filter function works as
## expected in RZ geometry. For this, we only use the last simulation timestep.

dim = "rz"
species_name = "electrons"

parser_filter_fn = "diags/diag_parser_filter000080"
parser_filter_expression = "(py-pz < 0) * (r<10e-6) * (z > 0)"
post_processing_utils.check_particle_filter(
    fn, parser_filter_fn, parser_filter_expression, dim, species_name
)

uniform_filter_fn = "diags/diag_uniform_filter000080"
uniform_filter_expression = "ids%3 == 0"
post_processing_utils.check_particle_filter(
    fn, uniform_filter_fn, uniform_filter_expression, dim, species_name
)

random_filter_fn = "diags/diag_random_filter000080"
random_fraction = 0.66
post_processing_utils.check_random_filter(
    fn, random_filter_fn, random_fraction, dim, species_name
)
