#!/usr/bin/env python3

# Copyright 2019 Jean-Luc Vay, Maxence Thevenet, Remi Lehe, Grant Johnson
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
import sys

import matplotlib.pyplot as plt
import yt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

yt.funcs.mylog.setLevel(50)

import numpy as np
from scipy.constants import c, e, epsilon_0, m_e

# this will be the name of the plot file
fn = sys.argv[1]

# Parameters (these parameters must match the parameters in `inputs.multi.rt`)
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

k = {"Ex": kx, "Ez": kz, "Jx": kx, "Jz": kz}
cos = {"Ex": (0, 1, 1), "Ez": (1, 1, 0), "Jx": (0, 1, 1), "Jz": (1, 1, 0)}
cos_rho = {"rho": (1, 1, 1)}


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


def get_theoretical_J_field(field, t):
    # wpdt/2 accounts for the Yee halfstep offset of the current
    dt = t / 40  # SPECIFIC to config parameters!
    amplitude = (
        -epsilon_0
        * wp
        * epsilon
        * (m_e * c**2 * k[field])
        / e
        * np.cos(wp * t - wp * dt / 2)
    )
    cos_flag = cos[field]
    x_contribution = get_contribution(cos_flag[0], kx)
    z_contribution = get_contribution(cos_flag[2], kz)

    J = amplitude * x_contribution[:, np.newaxis] * z_contribution[np.newaxis, :]

    return J


def get_theoretical_rho_field(field, t):
    amplitude = (
        epsilon_0 * epsilon * (m_e * c**2 * (kx * kx + kz * kz)) / e * np.sin(wp * t)
    )
    cos_flag = cos_rho[field]
    x_contribution = get_contribution(cos_flag[0], kx)
    z_contribution = get_contribution(cos_flag[2], kz)

    rho = amplitude * x_contribution[:, np.newaxis] * z_contribution[np.newaxis, :]

    return rho


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

# Check the validity of the currents
for field in ["Jx", "Jz"]:
    J_sim = data[("mesh", field)].to_ndarray()[:, :, 0]
    J_th = get_theoretical_J_field(field, t0)
    max_error = abs(J_sim - J_th).max() / abs(J_th).max()
    print("%s: Max error: %.2e" % (field, max_error))
    error_rel = max(error_rel, max_error)

# Check the validity of the charge
for field in ["rho"]:
    rho_sim = data[("boxlib", field)].to_ndarray()[:, :, 0]
    rho_th = get_theoretical_rho_field(field, t0)
    max_error = abs(rho_sim - rho_th).max() / abs(rho_th).max()
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
fig.savefig("Langmuir_fluid_multi_2d_analysis.png", dpi=200)

tolerance_rel = 0.05

print("error_rel    : " + str(error_rel))
print("tolerance_rel: " + str(tolerance_rel))

assert error_rel < tolerance_rel
