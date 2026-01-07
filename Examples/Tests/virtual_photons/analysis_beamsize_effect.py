#!/usr/bin/env python3

# Copyright 2025 Peter Kicsiny
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This test generates the population of virtual photons
# of two high-energy electrons.
# In one case the beam size effect (bse) is on and it is off for the other.
# Checks that when the bse is on, the virtual photons are correctly
# smeared within a disc of radius rho_max, and when the bse is off,
# the virtual photon coordinates are left as they are.

import os

import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, eV, m_e

m_e_ev = m_e * c**2 / eV


def rho_x(x):
    return hbar * c / (m_e_ev * x)


sigma_x = 1  # [m]
sigma_y = 1  # [m]
GeV = 1e9 * eV
energy = 182.5 * GeV  # [J]
gamma = 182.5 * 1e9 / m_e_ev
vphot_x_min = 1e-4 / gamma**2
hbar = 6.582119569e-16  # [eV s]
rho_max = rho_x(vphot_x_min)  # [m] c in numerator bc m is in unit of [ev/c^2]

diag_folder = "diags"
series = OpenPMDTimeSeries(os.path.join(diag_folder, "diag1"))

x_vphot_ele, y_vphot_ele, z_vphot_ele, uz_vphot_ele, w_vphot_ele = series.get_particle(
    ["x", "y", "z", "uz", "w"], species="virtual_photons1", iteration=1
)
x_vphot_pos, y_vphot_pos, z_vphot_pos, uz_vphot_pos, w_vphot_pos = series.get_particle(
    ["x", "y", "z", "uz", "w"], species="virtual_photons2", iteration=1
)

# convert to [Eb]
e_vphot_ele = uz_vphot_ele * c / energy
e_vphot_pos = -uz_vphot_pos * c / energy

x_ele, y_ele, z_ele, uz_ele, w_ele = series.get_particle(
    ["x", "y", "z", "uz", "w"], species="beam1", iteration=1
)
x_pos, y_pos, z_pos, uz_pos, w_pos = series.get_particle(
    ["x", "y", "z", "uz", "w"], species="beam2", iteration=1
)

########
# Plot #
########

plt.plot(x_vphot_pos, y_vphot_pos, "ro")
plt.plot(x_vphot_ele, y_vphot_ele, "bo")

plt.plot(x_ele, y_ele, "x", c="darkblue", markersize=10, label="e- BSE off")
plt.plot(x_pos, y_pos, "x", c="darkred", markersize=10, label="e+ BSE on")

phi_circ = np.linspace(0, 2 * np.pi, 100)
plt.plot(
    rho_max * np.cos(phi_circ) + x_pos,
    rho_max * np.sin(phi_circ) + y_pos,
    "r-",
    label=r"$\rho_{max}=\frac{\hbar}{m_e c x_{min}}$ [m]",
)

plt.xlabel("x [m]")
plt.ylabel("y [m]")

plt.axhline(512 * sigma_y)
plt.axhline(-512 * sigma_y)
plt.axvline(512 * sigma_x)
plt.axvline(-512 * sigma_x, label="WarpX grid boundaries")

plt.legend()

rr = np.sqrt(np.abs(x_vphot_pos - x_pos) ** 2 + np.abs(y_vphot_pos - y_pos) ** 2)
plt.hist(rr, bins=100, histtype="step", weights=np.ones_like(rr) / len(rr))

plt.axvline(rho_max, c="k", label=r"$\rho_{max}$")
plt.yscale("log")
plt.xlabel(r"$\rho$ [m]")
plt.ylabel("Count [1]")
plt.legend()

#########
# Tests #
#########

assert np.all(rr < rho_max)
assert np.all(x_vphot_ele == x_ele)
assert np.all(y_vphot_ele == y_ele)
assert ~np.all(x_vphot_pos == x_pos)
assert ~np.all(y_vphot_pos == y_pos)
