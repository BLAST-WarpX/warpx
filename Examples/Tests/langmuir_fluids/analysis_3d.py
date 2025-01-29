#!/usr/bin/env python3

# Copyright 2019-2022 Jean-Luc Vay, Maxence Thevenet, Remi Lehe, Axel Huebl, Grant Johnson
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
n_osc_y = 2
n_osc_z = 2
lo = [-20.0e-6, -20.0e-6, -20.0e-6]
hi = [20.0e-6, 20.0e-6, 20.0e-6]
Ncell = [64, 64, 64]

# Wave vector of the wave
kx = 2.0 * np.pi * n_osc_x / (hi[0] - lo[0])
ky = 2.0 * np.pi * n_osc_y / (hi[1] - lo[1])
kz = 2.0 * np.pi * n_osc_z / (hi[2] - lo[2])
# Plasma frequency
wp = np.sqrt((n * e**2) / (m_e * epsilon_0))

k = {"Ex": kx, "Ey": ky, "Ez": kz, "Jx": kx, "Jy": ky, "Jz": kz}
cos = {
    "Ex": (0, 1, 1),
    "Ey": (1, 0, 1),
    "Ez": (1, 1, 0),
    "Jx": (0, 1, 1),
    "Jy": (1, 0, 1),
    "Jz": (1, 1, 0),
}
cos_rho = {"rho": (1, 1, 1)}


def get_contribution(is_cos, k, idim):
    du = (hi[idim] - lo[idim]) / Ncell[idim]
    u = lo[idim] + du * (0.5 + np.arange(Ncell[idim]))
    if is_cos[idim] == 1:
        return np.cos(k * u)
    else:
        return np.sin(k * u)


def get_theoretical_field(field, t):
    amplitude = epsilon * (m_e * c**2 * k[field]) / e * np.sin(wp * t)
    cos_flag = cos[field]
    x_contribution = get_contribution(cos_flag, kx, 0)
    y_contribution = get_contribution(cos_flag, ky, 1)
    z_contribution = get_contribution(cos_flag, kz, 2)

    E = (
        amplitude
        * x_contribution[:, np.newaxis, np.newaxis]
        * y_contribution[np.newaxis, :, np.newaxis]
        * z_contribution[np.newaxis, np.newaxis, :]
    )

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
    x_contribution = get_contribution(cos_flag, kx, 0)
    y_contribution = get_contribution(cos_flag, ky, 1)
    z_contribution = get_contribution(cos_flag, kz, 2)

    J = (
        amplitude
        * x_contribution[:, np.newaxis, np.newaxis]
        * y_contribution[np.newaxis, :, np.newaxis]
        * z_contribution[np.newaxis, np.newaxis, :]
    )

    return J


def get_theoretical_rho_field(field, t):
    amplitude = (
        epsilon_0
        * epsilon
        * (m_e * c**2 * (kx * kx + ky * ky + kz * kz))
        / e
        * np.sin(wp * t)
    )
    cos_flag = cos_rho[field]
    x_contribution = get_contribution(cos_flag, kx, 0)
    y_contribution = get_contribution(cos_flag, ky, 1)
    z_contribution = get_contribution(cos_flag, kz, 2)

    rho = (
        amplitude
        * x_contribution[:, np.newaxis, np.newaxis]
        * y_contribution[np.newaxis, :, np.newaxis]
        * z_contribution[np.newaxis, np.newaxis, :]
    )

    return rho


# Read the file
ds = yt.load(fn)


t0 = ds.current_time.to_value()
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
edge = np.array(
    [
        (ds.domain_left_edge[2]).item(),
        (ds.domain_right_edge[2]).item(),
        (ds.domain_left_edge[0]).item(),
        (ds.domain_right_edge[0]).item(),
    ]
)

# Check the validity of the fields
error_rel = 0
for field in ["Ex", "Ey", "Ez"]:
    E_sim = data[("mesh", field)].to_ndarray()
    E_th = get_theoretical_field(field, t0)
    max_error = abs(E_sim - E_th).max() / abs(E_th).max()
    print("%s: Max error: %.2e" % (field, max_error))
    error_rel = max(error_rel, max_error)


# Check the validity of the currents
for field in ["Jx", "Jy", "Jz"]:
    J_sim = data[("mesh", field)].to_ndarray()
    J_th = get_theoretical_J_field(field, t0)
    max_error = abs(J_sim - J_th).max() / abs(J_th).max()
    print("%s: Max error: %.2e" % (field, max_error))
    error_rel = max(error_rel, max_error)

# Check the validity of the charge
for field in ["rho"]:
    rho_sim = data[("boxlib", field)].to_ndarray()
    rho_th = get_theoretical_rho_field(field, t0)
    max_error = abs(rho_sim - rho_th).max() / abs(rho_th).max()
    print("%s: Max error: %.2e" % (field, max_error))
    error_rel = max(error_rel, max_error)


# Plot the last field from the loop (Ez at iteration 40)
fig, (ax1, ax2) = plt.subplots(1, 2, dpi=100)
# First plot (slice at y=0)
E_plot = E_sim[:, Ncell[1] // 2 + 1, :]
vmin = E_plot.min()
vmax = E_plot.max()
cax1 = make_axes_locatable(ax1).append_axes("right", size="5%", pad="5%")
im1 = ax1.imshow(E_plot, origin="lower", extent=edge, vmin=vmin, vmax=vmax)
cb1 = fig.colorbar(im1, cax=cax1)
ax1.set_xlabel(r"$z$")
ax1.set_ylabel(r"$x$")
ax1.set_title(r"$E_z$ (sim)")
# Second plot (slice at y=0)
E_plot = E_th[:, Ncell[1] // 2 + 1, :]
vmin = E_plot.min()
vmax = E_plot.max()
cax2 = make_axes_locatable(ax2).append_axes("right", size="5%", pad="5%")
im2 = ax2.imshow(E_plot, origin="lower", extent=edge, vmin=vmin, vmax=vmax)
cb2 = fig.colorbar(im2, cax=cax2)
ax2.set_xlabel(r"$z$")
ax2.set_ylabel(r"$x$")
ax2.set_title(r"$E_z$ (theory)")
# Save figure
fig.tight_layout()
fig.savefig("Langmuir_fluid_multi_analysis.png", dpi=200)

tolerance_rel = 5e-2

print("error_rel    : " + str(error_rel))
print("tolerance_rel: " + str(tolerance_rel))

assert error_rel < tolerance_rel
