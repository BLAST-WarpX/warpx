#!/usr/bin/env python3

# Copyright 2019 Jean-Luc Vay, Maxence Thevenet, Remi Lehe
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from
# the script `inputs.multi.rt`. This simulates a 3D periodic plasma wave.
# The electric field in the simulation is given (in theory) by:
# $$ E_x = \epsilon \,\frac{m_e c^2 k_x}{q_e}\sin(k_x x)\cos(k_y y)\cos(k_z z)\sin( \omega_p t)$$
# $$ E_y = \epsilon \,\frac{m_e c^2 k_y}{q_e}\cos(k_x x)\sin(k_y y)\cos(k_z z)\sin( \omega_p t)$$
# $$ E_z = \epsilon \,\frac{m_e c^2 k_z}{q_e}\cos(k_x x)\cos(k_y y)\sin(k_z z)\sin( \omega_p t)$$
import os
import re
import sys

import matplotlib.pyplot as plt
import yt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

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

# Parse test name and check if particle_shape = 4 is used
particle_shape_4 = True if re.search("particle_shape_4", test_name) else False

# Parameters (must match the parameters in the inputs)
# FIXME read these parameters from warpx_used_inputs
epsilon = 0.01
n = 4.0e24
n_osc_x = 2
n_osc_z = 2
xmin = -20e-6
xmax = 20.0e-6
Nx = 128
zmin = -20e-6
zmax = 20.0e-6
Nz = 128

# Wave vector of the wave
kx = 2.0 * np.pi * n_osc_x / (xmax - xmin)
kz = 2.0 * np.pi * n_osc_z / (zmax - zmin)
# Plasma frequency
wp = np.sqrt((n * e**2) / (m_e * epsilon_0))

k = {"Ex": kx, "Ez": kz}
cos = {"Ex": (0, 1, 1), "Ez": (1, 1, 0)}


def get_contribution(is_cos, k):
    du = (xmax - xmin) / Nx
    u = xmin + du * (0.5 + np.arange(Nx))
    if is_cos == 1:
        return np.cos(k * u)
    else:
        return np.sin(k * u)


def get_theoretical_field(field, t):
    amplitude = epsilon * (m_e * c**2 * k[field]) / e * np.sin(wp * t)
    cos_flag = cos[field]
    x_contribution = get_contribution(cos_flag[0], kx)
    z_contribution = get_contribution(cos_flag[2], kz)

    E = amplitude * x_contribution[:, np.newaxis] * z_contribution[np.newaxis, :]

    return E


# Read the file
ds = yt.load(fn)
t0 = ds.current_time.to_value()
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
edge = np.array(
    [
        (ds.domain_left_edge[1]).item(),
        (ds.domain_right_edge[1]).item(),
        (ds.domain_left_edge[0]).item(),
        (ds.domain_right_edge[0]).item(),
    ]
)

# Check the validity of the fields
error_rel = 0
for field in ["Ex", "Ez"]:
    E_sim = data[("mesh", field)].to_ndarray()[:, :, 0]
    E_th = get_theoretical_field(field, t0)
    max_error = abs(E_sim - E_th).max() / abs(E_th).max()
    print("%s: Max error: %.2e" % (field, max_error))
    error_rel = max(error_rel, max_error)

# Plot the last field from the loop (Ez at iteration 40)
fig, (ax1, ax2) = plt.subplots(1, 2, dpi=100)
# First plot
vmin = E_sim.min()
vmax = E_sim.max()
cax1 = make_axes_locatable(ax1).append_axes("right", size="5%", pad="5%")
im1 = ax1.imshow(E_sim, origin="lower", extent=edge, vmin=vmin, vmax=vmax)
cb1 = fig.colorbar(im1, cax=cax1)
ax1.set_xlabel(r"$z$")
ax1.set_ylabel(r"$x$")
ax1.set_title(r"$E_z$ (sim)")
# Second plot
vmin = E_th.min()
vmax = E_th.max()
cax2 = make_axes_locatable(ax2).append_axes("right", size="5%", pad="5%")
im2 = ax2.imshow(E_th, origin="lower", extent=edge, vmin=vmin, vmax=vmax)
cb2 = fig.colorbar(im2, cax=cax2)
ax2.set_xlabel(r"$z$")
ax2.set_ylabel(r"$x$")
ax2.set_title(r"$E_z$ (theory)")
# Save figure
fig.tight_layout()
fig.savefig("Langmuir_multi_2d_analysis.png", dpi=200)

if particle_shape_4:
    # lower fidelity, due to smoothing
    tolerance_rel = 0.07
else:
    tolerance_rel = 0.05

print("error_rel    : " + str(error_rel))
print("tolerance_rel: " + str(tolerance_rel))

assert error_rel < tolerance_rel

# Check relative L-infinity spatial norm of rho/epsilon_0 - div(E)
# with current correction (and periodic single box option) or with Vay current deposition
if current_correction:
    tolerance = 1e-9
elif vay_deposition:
    tolerance = 1e-3
if current_correction or vay_deposition:
    rho = data[("boxlib", "rho")].to_ndarray()
    divE = data[("boxlib", "divE")].to_ndarray()
    error_rel = np.amax(np.abs(divE - rho / epsilon_0)) / np.amax(
        np.abs(rho / epsilon_0)
    )
    print("Check charge conservation:")
    print("error_rel = {}".format(error_rel))
    print("tolerance = {}".format(tolerance))
    assert error_rel < tolerance
