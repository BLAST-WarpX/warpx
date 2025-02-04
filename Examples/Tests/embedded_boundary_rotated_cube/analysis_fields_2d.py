#!/usr/bin/env python3

import sys

import numpy as np
import yt
from scipy.constants import c, mu_0, pi

# This is a script that analyses the simulation results from
# the script `inputs_3d`. This simulates a TMmnp mode in a PEC cubic resonator.
# The magnetic field in the simulation is given (in theory) by:
# $$ B_y = \mu \cos(k_x x)\cos(k_z z)\cos( \omega_p t)$$
# with
# $$ k_x = \frac{m\pi}{L}$$
# $$ k_y = \frac{n\pi}{L}$$
# $$ k_z = \frac{p\pi}{L}$$

hi = [0.8, 0.8]
lo = [-0.8, -0.8]
ncells = [32, 32]
dx = (hi[0] - lo[0]) / ncells[0]
dz = (hi[1] - lo[1]) / ncells[1]
m = 0
n = 1
Lx = 1.06
Lz = 1.06

# Open the right plot file
filename = sys.argv[1]
ds = yt.load(filename)
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
my_grid = ds.index.grids[0]

By_sim = my_grid["By"].squeeze().v

t = ds.current_time.to_value()

theta = np.pi / 8

# Compute the analytic solution
By_th = np.zeros(ncells)
for i in range(ncells[0]):
    for j in range(ncells[1]):
        x = i * dx + lo[0]
        z = j * dz + lo[1]
        xr = x * np.cos(-theta) + z * np.sin(-theta)
        zr = -x * np.sin(-theta) + z * np.cos(-theta)

        By_th[i, j] = (
            mu_0
            * (
                np.cos(m * pi / Lx * (xr - Lx / 2))
                * np.cos(n * pi / Lz * (zr - Lz / 2))
                * np.cos(np.pi / Lx * c * t)
            )
            * (By_sim[i, j] != 0)
        )

rel_tol_err = 1e-1

# Compute relative l^2 error on By
rel_err_y = np.sqrt(np.sum(np.square(By_sim - By_th)) / np.sum(np.square(By_th)))
assert rel_err_y < rel_tol_err
