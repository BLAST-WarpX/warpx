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
from analysis_utils import check_charge_conservation
from scipy.constants import c, e, epsilon_0, m_e

# test name
test_name = os.path.split(os.getcwd())[1]

# this will be the name of the plot file
fn = sys.argv[1]

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

# Plot the last field from the loop (Ez at iteration 40) and By, if available
# (By is zero in theory, useful to spot numerical artifacts)
panels = [(E_sim, r"$E_z$ (sim)"), (E_th, r"$E_z$ (theory)")]
if ("mesh", "By") in ds.derived_field_list:
    By_sim = data[("mesh", "By")].to_ndarray()[:, :, 0]
    panels.append((By_sim, r"$B_y$ (sim)"))
fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), dpi=100)
for ax, (F, title) in zip(axes, panels):
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad="5%")
    im = ax.imshow(
        F, origin="lower", extent=edge, vmin=F.min(), vmax=F.max(), cmap="RdBu_r"
    )
    fig.colorbar(im, cax=cax)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$x$")
    ax.set_title(title)
# Save figure
fig.tight_layout()
fig.savefig("Langmuir_multi_2d_analysis.png", dpi=200)

if particle_shape_4:
    # lower fidelity, due to smoothing
    tolerance_rel = 0.07
else:
    tolerance_rel = 0.0503

print("error_rel    : " + str(error_rel))
print("tolerance_rel: " + str(tolerance_rel))

assert error_rel < tolerance_rel

# Additional check on charge conservation for certain cases
# (e.g., Esirkepov or Vay deposition, current correction)
check_charge_conservation(data)
