#!/usr/bin/env python3

# Copyright 2025 Peter Kicsiny
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This test compares the output particle distribution from 2 simulations.
# In both simulations the bunch distribution is dumped after initialization.
# The seed is the same for both simulations.
# Sim 1: apply rotation by half crossing angle phi on bunch then dump.
# Sim 2: apply crabwaist operation then rotation then dump.
# In the test I apply the inverse rotation and CW on both bunches and
# compare them cordinate by coordinate.

import os

import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, eV, m_e

###############
# beam params #
###############

bunchint = 1.55e11
emit_x = 1e-6
emit_y = 1e-6
emit_z = 0
sigma_x = 1e-5  # [m]
sigma_y = 1e-5  # [m]
sigma_z = 1e-4  # [m]
sigma_px = emit_x / sigma_x  # [1]
sigma_py = emit_y / sigma_y  # [1]
sigma_pz = emit_z / sigma_z  # [1]

GeV = 1e9 * eV
energy = 182.5 * GeV  # [J]
m_e_ev = m_e * c**2 / eV
gamma = 182.5 * 1e9 / m_e_ev

phi = 1e-3  # [rad] half crossing angle

n_macroparts = int(1e6)

print(sigma_x, sigma_y, sigma_z, sigma_px, sigma_py, sigma_pz)

crabwaist_strength = 0.4  # cw strength [1]

###################
# load warpx data #
###################

# 2 simulations with same seed
# particles are shuffled so arrays wont be the same even with same seed

# sim 1: init a gaussian bunch and rotate by phi
series = OpenPMDTimeSeries(
    os.path.join("../test_3d_crabwaist_off_no_focusing", "diags", "diag_off")
)
(
    ux_ele_cw_0,
    uy_ele_cw_0,
    uz_ele_cw_0,
    w_ele_cw_0,
    x_ele_cw_0,
    y_ele_cw_0,
    z_ele_cw_0,
) = series.get_particle(
    ["ux", "uy", "uz", "w", "x", "y", "z"], species="beam1", iteration=0
)
(
    ux_pos_cw_0,
    uy_pos_cw_0,
    uz_pos_cw_0,
    w_pos_cw_0,
    x_pos_cw_0,
    y_pos_cw_0,
    z_pos_cw_0,
) = series.get_particle(
    ["ux", "uy", "uz", "w", "x", "y", "z"], species="beam2", iteration=0
)

# sim 2: init gaussian bunch, apply cw, rotate by phi
series = OpenPMDTimeSeries(
    os.path.join("../test_3d_crabwaist_on_no_focusing", "diags", "diag_on")
)
(
    ux_ele_cw_1,
    uy_ele_cw_1,
    uz_ele_cw_1,
    w_ele_cw_1,
    x_ele_cw_1,
    y_ele_cw_1,
    z_ele_cw_1,
) = series.get_particle(
    ["ux", "uy", "uz", "w", "x", "y", "z"], species="beam1", iteration=0
)
(
    ux_pos_cw_1,
    uy_pos_cw_1,
    uz_pos_cw_1,
    w_pos_cw_1,
    x_pos_cw_1,
    y_pos_cw_1,
    z_pos_cw_1,
) = series.get_particle(
    ["ux", "uy", "uz", "w", "x", "y", "z"], species="beam2", iteration=0
)

# sort all arrays according to this order
idx_ele_cw_0 = np.argsort(x_ele_cw_0)
idx_pos_cw_0 = np.argsort(x_pos_cw_0)
idx_ele_cw_1 = np.argsort(x_ele_cw_1)
idx_pos_cw_1 = np.argsort(x_pos_cw_1)

# first dataset, cw off
ux_ele_cw_0 = ux_ele_cw_0[idx_ele_cw_0]
uy_ele_cw_0 = uy_ele_cw_0[idx_ele_cw_0]
uz_ele_cw_0 = uz_ele_cw_0[idx_ele_cw_0]
w_ele_cw_0 = w_ele_cw_0[idx_ele_cw_0]
x_ele_cw_0 = x_ele_cw_0[idx_ele_cw_0]
y_ele_cw_0 = y_ele_cw_0[idx_ele_cw_0]
z_ele_cw_0 = z_ele_cw_0[idx_ele_cw_0]

ux_pos_cw_0 = ux_pos_cw_0[idx_pos_cw_0]
uy_pos_cw_0 = uy_pos_cw_0[idx_pos_cw_0]
uz_pos_cw_0 = uz_pos_cw_0[idx_pos_cw_0]
w_pos_cw_0 = w_pos_cw_0[idx_pos_cw_0]
x_pos_cw_0 = x_pos_cw_0[idx_pos_cw_0]
y_pos_cw_0 = y_pos_cw_0[idx_pos_cw_0]
z_pos_cw_0 = z_pos_cw_0[idx_pos_cw_0]

# second dataset, cw on
ux_ele_cw_1 = ux_ele_cw_1[idx_ele_cw_1]
uy_ele_cw_1 = uy_ele_cw_1[idx_ele_cw_1]
uz_ele_cw_1 = uz_ele_cw_1[idx_ele_cw_1]
w_ele_cw_1 = w_ele_cw_1[idx_ele_cw_1]
x_ele_cw_1 = x_ele_cw_1[idx_ele_cw_1]
y_ele_cw_1 = y_ele_cw_1[idx_ele_cw_1]
z_ele_cw_1 = z_ele_cw_1[idx_ele_cw_1]

ux_pos_cw_1 = ux_pos_cw_1[idx_pos_cw_1]
uy_pos_cw_1 = uy_pos_cw_1[idx_pos_cw_1]
uz_pos_cw_1 = uz_pos_cw_1[idx_pos_cw_1]
w_pos_cw_1 = w_pos_cw_1[idx_pos_cw_1]
x_pos_cw_1 = x_pos_cw_1[idx_pos_cw_1]
y_pos_cw_1 = y_pos_cw_1[idx_pos_cw_1]
z_pos_cw_1 = z_pos_cw_1[idx_pos_cw_1]

# after sorting it should be the same for both datasets
x_ele_pos_idx = x_ele_cw_0 > 0
x_pos_pos_idx = x_pos_cw_0 > 0

# sanity checks on sorting
assert np.all(x_ele_pos_idx == (x_ele_cw_1 > 0))
assert np.all(x_pos_pos_idx == (x_pos_cw_1 > 0))
assert np.all(uy_ele_cw_0 == uy_ele_cw_1)
assert np.all(uy_pos_cw_0 == uy_pos_cw_1)
assert np.all(x_ele_cw_0 == x_ele_cw_1)
assert np.all(x_pos_cw_0 == x_pos_cw_1)
assert np.all(z_ele_cw_0 == z_ele_cw_1)
assert np.all(z_pos_cw_0 == z_pos_cw_1)

####################
# plot phase space #
####################

# px is large bc it has been rotated
# e+ beam moves toward negative z
# R: clockwise rotation by phi half crossing angle in xz plane
# CW: crabwaist operation

# sim 1
fig, ax = plt.subplots(1, 2, figsize=(10, 3))

ax[0].plot(
    x_ele_cw_0[x_ele_pos_idx] / sigma_x,
    ux_ele_cw_0[x_ele_pos_idx] / sigma_px,
    "o",
    c="darkred",
    label=r"e- ($Rx>0$)",
)
ax[0].plot(
    x_ele_cw_0[~x_ele_pos_idx] / sigma_x,
    ux_ele_cw_0[~x_ele_pos_idx] / sigma_px,
    "o",
    c="darkblue",
    label=r"e- ($Rx<0$)",
)

ax[0].plot(
    x_pos_cw_0[x_pos_pos_idx] / sigma_x,
    ux_pos_cw_0[x_pos_pos_idx] / sigma_px,
    "rx",
    label=r"e+ ($Rx>0$)",
)
ax[0].plot(
    x_pos_cw_0[~x_pos_pos_idx] / sigma_x,
    ux_pos_cw_0[~x_pos_pos_idx] / sigma_px,
    "bx",
    label=r"e+ ($Rx<0$)",
)

ax[1].plot(
    y_ele_cw_0[x_ele_pos_idx] / sigma_y,
    uy_ele_cw_0[x_ele_pos_idx] / sigma_py,
    "o",
    c="darkred",
    label=r"e- ($Rx>0$)",
)
ax[1].plot(
    y_ele_cw_0[~x_ele_pos_idx] / sigma_y,
    uy_ele_cw_0[~x_ele_pos_idx] / sigma_py,
    "o",
    c="darkblue",
    label=r"e- ($Rx<0$)",
)

ax[1].plot(
    y_pos_cw_0[x_pos_pos_idx] / sigma_y,
    uy_pos_cw_0[x_pos_pos_idx] / sigma_py,
    "rx",
    label=r"e+ ($Rx>0$)",
)
ax[1].plot(
    y_pos_cw_0[~x_pos_pos_idx] / sigma_y,
    uy_pos_cw_0[~x_pos_pos_idx] / sigma_py,
    "bx",
    label=r"e+ ($Rx<0$)",
)

ax[0].set_xlabel(r"x [$\sigma_x$]")
ax[1].set_xlabel(r"y [$\sigma_y$]")
ax[0].set_ylabel(r"px [$\sigma_{px}$]")
ax[1].set_ylabel(r"py [$\sigma_{py}$]")

ax[0].legend()
ax[1].legend()

fig.suptitle(r"CW off ($CWRx$)")

# sim 2
fig, ax = plt.subplots(1, 2, figsize=(10, 3))

ax[0].plot(
    x_ele_cw_1[x_ele_pos_idx] / sigma_x,
    ux_ele_cw_1[x_ele_pos_idx] / sigma_px,
    "o",
    c="darkred",
    label=r"e- ($Rx>0$)",
)
ax[0].plot(
    x_ele_cw_1[~x_ele_pos_idx] / sigma_x,
    ux_ele_cw_1[~x_ele_pos_idx] / sigma_px,
    "o",
    c="darkblue",
    label=r"e- ($Rx<0$)",
)

ax[0].plot(
    x_pos_cw_1[x_pos_pos_idx] / sigma_x,
    ux_pos_cw_1[x_pos_pos_idx] / sigma_px,
    "rx",
    label=r"e+ ($Rx>0$)",
)
ax[0].plot(
    x_pos_cw_1[~x_pos_pos_idx] / sigma_x,
    ux_pos_cw_1[~x_pos_pos_idx] / sigma_px,
    "bx",
    label=r"e+ ($Rx<0$)",
)

ax[1].plot(
    y_ele_cw_1[x_ele_pos_idx] / sigma_y,
    uy_ele_cw_1[x_ele_pos_idx] / sigma_py,
    "o",
    c="darkred",
    label=r"e- ($Rx>0$)",
)
ax[1].plot(
    y_ele_cw_1[~x_ele_pos_idx] / sigma_y,
    uy_ele_cw_1[~x_ele_pos_idx] / sigma_py,
    "o",
    c="darkblue",
    label=r"e- ($Rx<0$)",
)

ax[1].plot(
    y_pos_cw_1[x_pos_pos_idx] / sigma_y,
    uy_pos_cw_1[x_pos_pos_idx] / sigma_py,
    "rx",
    label=r"e+ ($Rx>0$)",
)
ax[1].plot(
    y_pos_cw_1[~x_pos_pos_idx] / sigma_y,
    uy_pos_cw_1[~x_pos_pos_idx] / sigma_py,
    "bx",
    label=r"e+ ($Rx<0$)",
)

ax[0].set_xlabel(r"x [$\sigma_x$]")
ax[1].set_xlabel(r"y [$\sigma_y$]")
ax[0].set_ylabel(r"px [$\sigma_{px}$]")
ax[1].set_ylabel(r"py [$\sigma_{py}$]")

ax[0].legend()
ax[1].legend()

fig.suptitle(r"CW on ($Rx$)")


def rot(x, z, phi):
    """
    Anticlockwise rotation by phi
    To rotate clockwise use -phi
    Inside warpx inverse of this is used to rotate coords
    """
    c = np.cos(phi)
    s = np.sin(phi)
    x_rot = c * x - s * z
    z_rot = s * x + c * z
    return x_rot, z_rot


############################################
# apply inverse rotation on cw off dataset #
############################################

# beam 1 rotate anticlockwise by phi
x_ele_cw_0_init, z_ele_cw_0_init = rot(x_ele_cw_0, z_ele_cw_0, phi)
ux_ele_cw_0_init, uz_ele_cw_0_init = rot(ux_ele_cw_0, uz_ele_cw_0, phi)
y_ele_cw_0_init = y_ele_cw_0
uy_ele_cw_0_init = uy_ele_cw_0

# beam 2 rotate clockwise by phi
x_pos_cw_0_init, z_pos_cw_0_init = rot(x_pos_cw_0, z_pos_cw_0, -phi)
ux_pos_cw_0_init, uz_pos_cw_0_init = rot(ux_pos_cw_0, uz_pos_cw_0, -phi)
y_pos_cw_0_init = y_pos_cw_0
uy_pos_cw_0_init = uy_pos_cw_0

##########################################################
# apply inverse rotation and inverse cw on cw on dataset #
##########################################################

# beam 1 rotate anticlockwise by phi
x_ele_cw_1_rot, z_ele_cw_1_rot = rot(x_ele_cw_1, z_ele_cw_1, phi)
ux_ele_cw_1_rot, uz_ele_cw_1_rot = rot(ux_ele_cw_1, uz_ele_cw_1, phi)

# beam 2 rotate clockwise by phi
x_pos_cw_1_rot, z_pos_cw_1_rot = rot(x_pos_cw_1, z_pos_cw_1, -phi)
ux_pos_cw_1_rot, uz_pos_cw_1_rot = rot(ux_pos_cw_1, uz_pos_cw_1, -phi)

acw_ele = -crabwaist_strength / np.tan(2 * phi)
acw_pos = -crabwaist_strength / np.tan(2 * -phi)

# beam 1 apply inverse cw
z_ele_cw_1_init = z_ele_cw_1_rot
uz_ele_cw_1_init = uz_ele_cw_1_rot
x_ele_cw_1_init = x_ele_cw_1_rot
uy_ele_cw_1_init = uy_ele_cw_1

ux_ele_cw_1_init = ux_ele_cw_1_rot - 0.5 * acw_ele * uy_ele_cw_1_init**2
y_ele_cw_1_init = y_ele_cw_1 + acw_ele * x_ele_cw_1_init * uy_ele_cw_1_init

# beam 2 apply inverse cw
z_pos_cw_1_init = z_pos_cw_1_rot
uz_pos_cw_1_init = uz_pos_cw_1_rot
x_pos_cw_1_init = x_pos_cw_1_rot
uy_pos_cw_1_init = uy_pos_cw_1

ux_pos_cw_1_init = ux_pos_cw_1_rot - 0.5 * acw_pos * uy_pos_cw_1_init**2
y_pos_cw_1_init = y_pos_cw_1 + acw_pos * x_pos_cw_1_init * uy_pos_cw_1_init

####################
# plot phase space #
####################

# sim 2 backtransformed
fig, ax = plt.subplots(1, 2, figsize=(10, 3))

ax[0].plot(
    x_ele_cw_1_init[x_ele_pos_idx] / sigma_x,
    ux_ele_cw_1_init[x_ele_pos_idx] / sigma_px,
    "o",
    c="darkred",
    label=r"e- ($Rx>0$)",
)
ax[0].plot(
    x_ele_cw_1_init[~x_ele_pos_idx] / sigma_x,
    ux_ele_cw_1_init[~x_ele_pos_idx] / sigma_px,
    "o",
    c="darkblue",
    label=r"e- ($Rx<0$)",
)

ax[0].plot(
    x_pos_cw_1_init[x_pos_pos_idx] / sigma_x,
    ux_pos_cw_1_init[x_pos_pos_idx] / sigma_px,
    "rx",
    label=r"e+ ($Rx>0$)",
)
ax[0].plot(
    x_pos_cw_1_init[~x_pos_pos_idx] / sigma_x,
    ux_pos_cw_1_init[~x_pos_pos_idx] / sigma_px,
    "bx",
    label=r"e+ ($Rx<0$)",
)

ax[1].plot(
    y_ele_cw_1_init[x_ele_pos_idx] / sigma_y,
    uy_ele_cw_1_init[x_ele_pos_idx] / sigma_py,
    "o",
    c="darkred",
    label=r"e- ($Rx>0$)",
)
ax[1].plot(
    y_ele_cw_1_init[~x_ele_pos_idx] / sigma_y,
    uy_ele_cw_1_init[~x_ele_pos_idx] / sigma_py,
    "o",
    c="darkblue",
    label=r"e- ($Rx<0$)",
)

ax[1].plot(
    y_pos_cw_1_init[x_pos_pos_idx] / sigma_y,
    uy_pos_cw_1_init[x_pos_pos_idx] / sigma_py,
    "rx",
    label=r"e+ ($Rx>0$)",
)
ax[1].plot(
    y_pos_cw_1_init[~x_pos_pos_idx] / sigma_y,
    uy_pos_cw_1_init[~x_pos_pos_idx] / sigma_py,
    "bx",
    label=r"e+ ($Rx<0$)",
)

ax[0].set_xlabel(r"x [$\sigma_x$]")
ax[1].set_xlabel(r"y [$\sigma_y$]")
ax[0].set_ylabel(r"px [$\sigma_{px}$]")
ax[1].set_ylabel(r"py [$\sigma_{py}$]")

ax[0].legend()
ax[1].legend()

fig.suptitle(r"CW on backtransformed ($CW^{-1}R^{-1}x$)")

#########
# tests #
#########

# these are trivial
assert np.all(x_ele_cw_1_init == x_ele_cw_0_init)
assert np.all(x_pos_cw_1_init == x_pos_cw_0_init)

assert np.all(uy_ele_cw_1_init == uy_ele_cw_0_init)
assert np.all(uy_pos_cw_1_init == uy_pos_cw_0_init)

assert np.all(z_ele_cw_1_init == z_ele_cw_0_init)
assert np.all(z_pos_cw_1_init == z_pos_cw_0_init)

assert np.allclose(
    uz_ele_cw_1_init / gamma, uz_ele_cw_0_init / gamma, rtol=1e-14, atol=1e-14
)
assert np.allclose(
    uz_pos_cw_1_init / gamma, uz_pos_cw_0_init / gamma, rtol=1e-14, atol=1e-14
)

# check the variables that are actually modified by crab waist
assert np.allclose(ux_ele_cw_1_init, ux_ele_cw_0_init, rtol=1e-6, atol=1e-12)
assert np.allclose(ux_pos_cw_1_init, ux_pos_cw_0_init, rtol=1e-6, atol=1e-12)

assert np.allclose(y_ele_cw_1_init, y_ele_cw_0_init, rtol=1e-8, atol=1e-17)
assert np.allclose(y_pos_cw_1_init, y_pos_cw_0_init, rtol=1e-8, atol=1e-17)
