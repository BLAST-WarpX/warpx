#!/usr/bin/env python3

# Copyright 2024 Arianna Formenti
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, eV, m_e, micro, milli

sigmax = 2 * micro
sigmay = 1 * micro
sigmaz = 3.0 * micro
emitx = 100 * milli
emity = 20 * milli
gamma = 125 * 1e9 * eV / (m_e * c**2)
x_m = 10 * sigmax
y_m = 10 * sigmay
z_m = 10 * sigmaz
focal_distance = 4 * sigmaz
theta = 0.25 * np.pi


def s(z, sigma, emit):
    """The theoretical size of a focusing beam (in the absence of space charge),
    at position z, given its emittance and size at focus."""
    return np.sqrt(sigma**2 + emit**2 * (z - z_m - focal_distance) ** 2 / (sigma**2))


series = OpenPMDTimeSeries("./diags/openpmd/")

# Rotated beam
xrot, yrot, zrot, w, uxrot, uyrot, uzrot = series.get_particle(
    ["x", "y", "z", "w", "ux", "uy", "uz"], species="beam1", iteration=0, plot=False
)

# Rotate the beam back so that it propagates along z
x = x_m + np.cos(-theta) * (xrot - x_m) + np.sin(-theta) * (zrot - z_m)
y = yrot
z = z_m - np.sin(-theta) * (xrot - x_m) + np.cos(-theta) * (zrot - z_m)


# Compute the size of the beam in different z slices
gridz = np.linspace(z_m - focal_distance * 0.8, z_m + focal_distance * 0.8, 64)
tol = 0.5 * (gridz[1] - gridz[0])
sx, sy = [], []
for z_g in gridz:
    i = np.abs(z - z_g) < tol
    if np.sum(i) != 0:
        mux = np.average(x[i], weights=w[i])
        muy = np.average(y[i], weights=w[i])
        sx.append(np.sqrt(np.average((x[i] - mux) ** 2, weights=w[i])))
        sy.append(np.sqrt(np.average((y[i] - muy) ** 2, weights=w[i])))
    else:
        sx.append(0.0)
        sy.append(0.0)

# Theoretical prediction for the size of the beam in each z slice
sx_theory = s(gridz, sigmax, emitx / gamma)
sy_theory = s(gridz, sigmay, emity / gamma)

assert np.allclose(sx, sx_theory, rtol=0.086, atol=0)
assert np.allclose(sy, sy_theory, rtol=0.056, atol=0)

# Rotate the beam back so that it propagates along z
ux = np.cos(-theta) * uxrot + np.sin(-theta) * uzrot
uy = uyrot
uz = -np.sin(-theta) * uxrot + np.cos(-theta) * uzrot


uz_m = np.average(uz, weights=w)
uz_th = np.sqrt(np.average((uz - uz_m) ** 2, weights=w))

ux_m = np.average(ux, weights=w)
ux_th = np.sqrt(np.average((ux - ux_m) ** 2, weights=w))


assert np.isclose(uz_m, gamma, rtol=1e-8, atol=0)
assert np.abs(uz_th) < 1e-8
assert np.abs(ux_m) < 4
assert np.isclose(ux_th, emitx / sigmax, rtol=1e-3, atol=0)
