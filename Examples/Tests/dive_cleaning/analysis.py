#!/usr/bin/env python3

# Copyright 2019-2020 Axel Huebl, Remi Lehe
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
This script checks the dive cleaning and PML routines, by initializing a
Gaussian beam in the center of the simulation box, and let the error in
space-charge field propagate away and be absorbed in the PML.

This script verifies that the field at the end of the simulation corresponds
to the theoretical field of a Gaussian beam.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as scc
import yt
from scipy.special import gammainc

yt.funcs.mylog.setLevel(0)

# Parameters from the Simulation
Qtot = -1.0e-20
r0 = 2.0e-6

# Open data file
filename = sys.argv[1]
ds = yt.load(filename)
# yt 4.0+ has rounding issues with our domain data:
# RuntimeError: yt attempted to read outside the boundaries
#               of a non-periodic domain along dimension 0.
if "force_periodicity" in dir(ds):
    ds.force_periodicity()

# Extract data
ad0 = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
Ex_array = ad0[("mesh", "Ex")].to_ndarray().squeeze()
if ds.dimensionality == 2:
    # Rename the z dimension as y, so as to make this script work for 2d and 3d
    Ey_array = ad0[("mesh", "Ez")].to_ndarray().squeeze()
    E_array = (Ex_array**2 + Ey_array**2) ** 0.5
    relative_tolerance = 0.1
elif ds.dimensionality == 3:
    Ey_array = ad0[("mesh", "Ey")].to_ndarray()
    Ez_array = ad0[("mesh", "Ez")].to_ndarray()
    E_array = (Ex_array**2 + Ey_array**2 + Ez_array**2) ** 0.5
    relative_tolerance = 0.165

# Extract grid coordinates
Nx, Ny, Nz = ds.domain_dimensions
xmin, ymin, zmin = ds.domain_left_edge.v
Lx, Ly, Lz = ds.domain_width.v
x = xmin + Lx / Nx * (0.5 + np.arange(Nx))
y = ymin + Ly / Ny * (0.5 + np.arange(Ny))
z = zmin + Lz / Nz * (0.5 + np.arange(Nz))

# Compute theoretical field
if ds.dimensionality == 2:
    x_2d, y_2d = np.meshgrid(x, y, indexing="ij")
    r2 = x_2d**2 + y_2d**2
    factor = (
        (Qtot / r0) / (2 * np.pi * scc.epsilon_0 * r2) * (1 - np.exp(-r2 / (2 * r0**2)))
    )
    Ex_th = x_2d * factor
    Ey_th = y_2d * factor
    E_th = (Ex_th**2 + Ey_th**2) ** 0.5
elif ds.dimensionality == 3:
    x_2d, y_2d, z_2d = np.meshgrid(x, y, z, indexing="ij")
    r2 = x_2d**2 + y_2d**2 + z_2d**2
    factor = (
        Qtot
        / (4 * np.pi * scc.epsilon_0 * r2**1.5)
        * gammainc(3.0 / 2, r2 / (2.0 * r0**2))
    )
    Ex_th = factor * x_2d
    Ey_th = factor * y_2d
    Ez_th = factor * z_2d
    E_th = (Ex_th**2 + Ey_th**2 + Ez_th**2) ** 0.5


# Plot theory and data
def make_2d(arr):
    if arr.ndim == 3:
        return arr[:, :, Nz // 2]
    else:
        return arr


plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.title("E: Theory")
plt.imshow(make_2d(E_th))
plt.colorbar()
plt.subplot(222)
plt.title("E: Simulation")
plt.imshow(make_2d(E_array))
plt.colorbar()
plt.subplot(223)
plt.title("E: Diff")
plt.imshow(make_2d(E_th - E_array))
plt.colorbar()
plt.subplot(224)
plt.title("E: Relative diff")
plt.imshow(make_2d((E_th - E_array) / E_th))
plt.colorbar()
plt.savefig("Comparison.png")


# Automatically check the results
def check(E, E_th, label):
    print("Relative error in %s: %.3f" % (label, abs(E - E_th).max() / E_th.max()))
    assert np.allclose(E, E_th, atol=relative_tolerance * E_th.max())


check(Ex_array, Ex_th, "Ex")
check(Ey_array, Ey_th, "Ey")
if ds.dimensionality == 3:
    check(Ez_array, Ez_th, "Ez")
