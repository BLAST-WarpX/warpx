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
ncells = [32, 32, 1]
dx = (hi[0] - lo[0]) / ncells[0]
dz = (hi[1] - lo[1]) / ncells[1]
m = 0
n = 1
Lx = 1
Lz = 1

# Open the right plot file
filename = sys.argv[1]
ds = yt.load(filename)
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

t = ds.current_time.to_value()

# Compute the analytic solution
By_th = np.zeros(ncells)
for i in range(ncells[0]):
    for j in range(ncells[1]):
        x = (i + 0.5) * dx + lo[0]
        z = (j + 0.5) * dz + lo[1]

        By_th[i, j, 0] = mu_0 * (
            np.cos(m * pi / Lx * (x - Lx / 2))
            * np.cos(n * pi / Lz * (z - Lz / 2))
            * (-Lx / 2 <= x < Lx / 2)
            * (-Lz / 2 <= z < Lz / 2)
            * np.cos(np.pi / Lx * c * t)
        )
rel_tol_err = 1e-3

# Compute relative l^2 error on By
By_sim = data["By"].to_ndarray()
rel_err_y = np.sqrt(np.sum(np.square(By_sim - By_th)) / np.sum(np.square(By_th)))
assert rel_err_y < rel_tol_err

# Compute relative l^2 error on Ey
Ey_sim = data["Ey"].to_ndarray()
rel_err_y = np.sqrt(np.sum(np.square(Ey_sim / c - By_th)) / np.sum(np.square(By_th)))
