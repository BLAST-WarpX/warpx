#!/usr/bin/env python3

# Copyright 2026
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This script analyses the 2D macroscopic-solver regression test.
#
# A standing EM cavity mode is initialized in a PEC box in the (x, z) plane
# through the out-of-plane field By(x, z) = cos(kx x) cos(kz z) with E = 0 at
# t = 0, where kx = m*pi/Lx and kz = p*pi/Lz. In a non-dispersive dielectric
# (epsilon_r = 1.5, mu = mu0, sigma = 0) the mode oscillates as
#     By(x, z, t) = cos(kx x) cos(kz z) cos(omega t)
# with omega = c * sqrt(kx^2 + kz^2) / sqrt(epsilon_r). The reduced frequency
# (slower than vacuum by 1/sqrt(epsilon_r)) is the signature of the macroscopic
# solver and is only reproduced if the 2D macroscopic E-update is correct.

import sys

import numpy as np
import yt
from scipy.constants import c, pi

yt.funcs.mylog.setLevel(0)

# Test parameters (must match the inputs file)
lo = [0.0, 0.0]
hi = [1.0, 1.0]
Lx = 1.0
Lz = 1.0
m = 1
p = 1
kx = m * pi / Lx
kz = p * pi / Lz
epsilon_r = 1.5

# Open the plot file
filename = sys.argv[1]
ds = yt.load(filename)
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

# In 2D the array axes are (x, z)
By_sim = data[("boxlib", "By")].v.squeeze()
nx, nz = By_sim.shape
dx = (hi[0] - lo[0]) / nx
dz = (hi[1] - lo[1]) / nz

# Cell coordinates (the half-cell staggering offset is negligible at this
# resolution: kx * dx / 2, kz * dz / 2 << 1, well within the test tolerance)
x = lo[0] + (np.arange(nx) + 0.5) * dx
z = lo[1] + (np.arange(nz) + 0.5) * dz
X, Z = np.meshgrid(x, z, indexing="ij")

# Mode frequency, reduced by the dielectric
omega = c * np.sqrt(kx**2 + kz**2) / np.sqrt(epsilon_r)
t = ds.current_time.to_value()

# Analytic standing-wave solution
By_th = np.cos(kx * X) * np.cos(kz * Z) * np.cos(omega * t)

# Relative l2 error
rel_tol_err = 1e-1
rel_err = np.sqrt(np.sum(np.square(By_sim - By_th)) / np.sum(np.square(By_th)))
print(
    f"t = {t:.6e} s, omega*t = {omega * t:.4f} rad, rel. l2 error on By = {rel_err:.4e}"
)
assert rel_err < rel_tol_err
