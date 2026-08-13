#!/usr/bin/env python3

# Copyright 2025 Peter Kicsiny
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, eV, m_e

do_plots = True

sigma_x = 1e-5  # [m]
sigma_y = 1e-5  # [m]
sigma_z = 1e-4  # [m]
gamma = 182.5e9 * eV / (m_e * c**2)
phi = 1e-3  # [rad]
focal_distance = 4 * sigma_z
crabwaist_strength = 0.4

beams = {
    "beam1": (0.1 * sigma_x, -0.2 * sigma_y, gamma, phi, False),
    "beam2": (-0.1 * sigma_x, 0.2 * sigma_y, -gamma, -phi, True),
}

series = OpenPMDTimeSeries("./diags/diag")

if do_plots:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

for i, (species, (x_m, y_m, uz_m, angle, rotate_momenta)) in enumerate(beams.items()):
    ux, uy, uz, x, y, z = series.get_particle(
        ["ux", "uy", "uz", "x", "y", "z"], species=species, iteration=0
    )

    # Undo the rotation around the beam centroid.
    dx = x - x_m
    z, x = (
        np.cos(-angle) * z - np.sin(-angle) * dx,
        x_m + np.sin(-angle) * z + np.cos(-angle) * dx,
    )

    if rotate_momenta:
        ux_old = ux.copy()
        uz, ux = (
            np.cos(-angle) * uz - np.sin(-angle) * ux,
            np.sin(-angle) * uz + np.cos(-angle) * ux,
        )

    # Undo focusing. Beam 2 travels toward negative z.
    dz = np.sign(uz_m) * focal_distance - z
    x_cw = x + ux / uz * dz
    y_cw = y + uy / uz * dz

    alpha = -crabwaist_strength / np.tan(2 * angle)
    delta_ux = 0.5 * alpha * uy**2 / abs(uz_m)
    delta_y = -alpha * (x_cw - x_m) * uy / abs(uz_m)

    np.testing.assert_allclose(ux, delta_ux, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(y_cw, y_m + delta_y)

    if do_plots:
        x_plot = (x_cw - x_m) * 1e6
        axes[0, i].scatter(x_plot, (y_cw - delta_y - y_m) * 1e9, s=5, label="No CW")
        axes[0, i].scatter(x_plot, (y_cw - y_m) * 1e9, s=5, label="CW")
        axes[0, i].set(xlabel="x-x_m [um]", ylabel="y-y_m [nm]", title=species)
        axes[1, i].scatter(uy, (ux - delta_ux) * 1e6, s=5, label="No CW")
        axes[1, i].scatter(uy, ux * 1e6, s=5, label="CW")
        axes[1, i].set(xlabel="uy", ylabel="ux [1e-6]")

if do_plots:
    for ax in axes.flat:
        ax.legend()
    fig.tight_layout()
    fig.savefig("crabwaist.png", dpi=200)
