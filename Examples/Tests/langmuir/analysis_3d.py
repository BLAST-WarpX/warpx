#!/usr/bin/env python3

# Copyright 2019-2022 Jean-Luc Vay, Maxence Thevenet, Remi Lehe, Axel Huebl
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

# Parse test name and check if div(E)/div(B) cleaning (warpx.do_div<e,b>_cleaning=1) is used
div_cleaning = True if re.search("div_cleaning", test_name) else False

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

k = {"Ex": kx, "Ey": ky, "Ez": kz}
cos = {"Ex": (0, 1, 1), "Ey": (1, 0, 1), "Ez": (1, 1, 0)}


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


# Read the file
ds = yt.load(fn)

# Check that the particle selective output worked:
species = "electrons"
print("ds.field_list", ds.field_list)
for field in ["particle_weight", "particle_momentum_x"]:
    print("assert that this is in ds.field_list", (species, field))
    assert (species, field) in ds.field_list
for field in ["particle_momentum_y", "particle_momentum_z"]:
    print("assert that this is NOT in ds.field_list", (species, field))
    assert (species, field) not in ds.field_list
species = "positrons"
for field in ["particle_momentum_x", "particle_momentum_y"]:
    print("assert that this is NOT in ds.field_list", (species, field))
    assert (species, field) not in ds.field_list

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
fig.savefig("Langmuir_multi_analysis.png", dpi=200)

tolerance_rel = 5e-2

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

if div_cleaning:
    ds_old = yt.load("diags/diag1000038")
    ds_mid = yt.load("diags/diag1000039")
    ds_new = yt.load(fn)  # this is the last plotfile

    ad_old = ds_old.covering_grid(
        level=0, left_edge=ds_old.domain_left_edge, dims=ds_old.domain_dimensions
    )
    ad_mid = ds_mid.covering_grid(
        level=0, left_edge=ds_mid.domain_left_edge, dims=ds_mid.domain_dimensions
    )
    ad_new = ds_new.covering_grid(
        level=0, left_edge=ds_new.domain_left_edge, dims=ds_new.domain_dimensions
    )

    rho = ad_mid["rho"].v.squeeze()
    divE = ad_mid["divE"].v.squeeze()
    F_old = ad_old["F"].v.squeeze()
    F_new = ad_new["F"].v.squeeze()

    # Check max norm of error on dF/dt = div(E) - rho/epsilon_0
    # (the time interval between the old and new data is 2*dt)
    dt = 1.203645751e-15
    x = F_new - F_old
    y = (divE - rho / epsilon_0) * 2 * dt
    error_rel = np.amax(np.abs(x - y)) / np.amax(np.abs(y))
    tolerance = 1e-2
    print("Check div(E) cleaning:")
    print("error_rel = {}".format(error_rel))
    print("tolerance = {}".format(tolerance))
    assert error_rel < tolerance
