#!/usr/bin/env python3


import sys

import matplotlib.pyplot as plt
import numpy as np
import yt
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, e

yt.funcs.mylog.setLevel(0)

filename = sys.argv[1]
ts = OpenPMDTimeSeries(filename)

# Plot the ion beam and electric potential at iteration `iteration`.
# Also checks if particle energies are within relative tolerance of target_energy_keV.
iteration = 1000

eb_covered, info = ts.get_field("eb_covered", iteration=iteration, slice_across="y")
phi, info = ts.get_field("phi", iteration=iteration, slice_across="y")
plt.subplot(2, 1, 1)
extent = np.concatenate((info.imshow_extent[2:], info.imshow_extent[:2]))
plt.imshow(
    phi.T,
    cmap="plasma_r",
    vmin=-40e3,
    vmax=0,
    aspect="auto",
    interpolation="bilinear",
    extent=1e3 * extent,
    origin="lower",
    alpha=0.7,
)

# Plot ions
xp, zp, uxp, uyp, uzp, mp = ts.get_particle(
    ["x", "z", "ux", "uy", "uz", "mass"], species="Dplus", iteration=iteration
)
plt.plot(1e3 * zp, 1e3 * xp, "r.", ms=0.8)
plt.ylabel("x [mm]")
plt.xticks(2 * np.arange(12))
plt.grid()

# Plot contours
phi_levels = list(
    [-41e3 + i * 0.3e3 for i in range(1, 4)]
    + [-34e3 + i * 5e3 for i in range(7)]
    + [-1e3 + i * 0.3e3 for i in range(1, 5)]
)
plt.contour(
    phi.T, extent=1e3 * extent, levels=phi_levels, linewidths=0.5, colors="black"
)
plt.contour(eb_covered.T, extent=1e3 * extent, levels=[0.8], linewidths=2)

# Plot kinetic energy
plt.subplot(2, 1, 2)
energy_keV = 0.5 * mp * c * c * (uxp**2 + uyp**2 + uzp**2) / e / 1e3
plt.plot(1e3 * zp, energy_keV, "r.", ms=0.8)
plt.xlabel("z [mm]")
plt.ylabel("Kinetic energy [keV]")
plt.xticks(2 * np.arange(12))
plt.grid()
plt.ylim(0, 50)
plt.xlim(-2, 23)

mask = (zp * 1e3 >= 14) & (zp * 1e3 <= 23)  # zp*1e3 [mm]
target_energy_keV = 40  # kEv
rel_error_energy = np.abs(energy_keV[mask] - target_energy_keV) / target_energy_keV
tolerance = 0.05

assert np.all(rel_error_energy < tolerance), (
    "Particle energy tails is NOT within the relative tolerance of target_energy_keV!"
)
