#!/usr/bin/env python3

# Copyright 2024 Arianna Formenti
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, eV, m_e, micro, nano

GeV = 1e9 * eV
energy = 125.0 * GeV
gamma = energy / (m_e * c**2)
sigmax = 516.0 * nano
sigmay = 7.7 * nano
sigmaz = 300.0 * micro
nz = 256
Lz = 20 * sigmaz
gridz = np.linspace(-0.5 * Lz, 0.5 * Lz, nz)
tol = gridz[1] - gridz[0]
emitx = 50 * micro
emity = 20 * nano
focal_distance = 4 * sigmaz


def s(z, sigma0, emit):
    """The theoretical size of a focusing beam (in the absence of space charge),
    at position z, given its emittance and size at focus."""
    return np.sqrt(sigma0**2 + emit**2 * (z - focal_distance) ** 2 / sigma0**2)


ts = OpenPMDTimeSeries("./diags/openpmd/")

(
    x,
    y,
    z,
    w,
) = ts.get_particle(["x", "y", "z", "w"], species="beam1", iteration=0, plot=False)

imin = np.argmin(np.sqrt((gridz + 0.8 * focal_distance) ** 2))
imax = np.argmin(np.sqrt((gridz - 0.8 * focal_distance) ** 2))

sx, sy = [], []
# Compute the size of the beam in each z slice
subgrid = gridz[imin:imax]
for d in subgrid:
    i = np.sqrt((z - d) ** 2) < tol
    if np.sum(i) != 0:
        mux = np.average(x[i], weights=w[i])
        muy = np.average(y[i], weights=w[i])
        sx.append(np.sqrt(np.average((x[i] - mux) ** 2, weights=w[i])))
        sy.append(np.sqrt(np.average((y[i] - muy) ** 2, weights=w[i])))

# Theoretical prediction for the size of the beam in each z slice
sx_theory = s(subgrid, sigmax, emitx / gamma)
sy_theory = s(subgrid, sigmay, emity / gamma)

assert np.allclose(sx, sx_theory, rtol=0.051, atol=0)
assert np.allclose(sy, sy_theory, rtol=0.038, atol=0)
