#!/usr/bin/env python3

# Copyright 2026
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This script analyses the 1D macroscopic-solver regression test.
#
# A standing EM cavity mode is initialized between two PEC walls along z through
# the external field By(z) = cos(k z) with E = 0 at t = 0, where k = p*pi/L.
# In a non-dispersive dielectric (epsilon_r = 1.5, mu = mu0, sigma = 0) the mode
# oscillates as
#     By(z, t) = cos(k z) cos(omega t),   omega = c * k / sqrt(epsilon_r)
# The reduced frequency omega (slower than vacuum by 1/sqrt(epsilon_r)) is the
# signature of the macroscopic solver: it is only reproduced if the 1D
# macroscopic E-update is correct.

import sys

import numpy as np
import yt
from scipy.constants import c, pi

yt.funcs.mylog.setLevel(0)

# Test parameters (must match the inputs file)
lo = 0.0
hi = 1.0
L = 1.0
p = 1
kz = p * pi / L
epsilon_r = 1.5

# Open the plot file
filename = sys.argv[1]
ds = yt.load(filename)
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

By_sim = data[("boxlib", "By")].v.squeeze()
ncells = By_sim.size
dz = (hi - lo) / ncells

# Cell coordinates (the half-cell staggering offset is negligible at this
# resolution: kz * dz / 2 << 1, well within the test tolerance)
z = lo + (np.arange(ncells) + 0.5) * dz

# Mode frequency, reduced by the dielectric
omega = c * kz / np.sqrt(epsilon_r)
t = ds.current_time.to_value()

# Analytic standing-wave solution
By_th = np.cos(kz * z) * np.cos(omega * t)

# Relative l2 error
rel_tol_err = 1e-1
rel_err = np.sqrt(np.sum(np.square(By_sim - By_th)) / np.sum(np.square(By_th)))
print(
    f"t = {t:.6e} s, omega*t = {omega * t:.4f} rad, rel. l2 error on By = {rel_err:.4e}"
)
assert rel_err < rel_tol_err
