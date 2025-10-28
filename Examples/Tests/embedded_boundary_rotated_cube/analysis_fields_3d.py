#!/usr/bin/env python3

# Copyright 2021 Lorenzo Giacomel
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


import sys

import numpy as np
import yt
from scipy.constants import c, mu_0, pi

# This is a script that analyses the simulation results from
# the script `inputs_3d`. This simulates a TMmnp mode in a PEC cubic resonator rotated by pi/8.
# The magnetic field in the simulation is given (in theory) by:
# $$ B_x = \frac{-2\mu}{h^2}\, k_x k_z \sin(k_x x)\cos(k_y y)\cos(k_z z)\cos( \omega_p t)$$
# $$ B_y = \frac{-2\mu}{h^2}\, k_y k_z \cos(k_x x)\sin(k_y y)\cos(k_z z)\cos( \omega_p t)$$
# $$ B_z = \cos(k_x x)\cos(k_y y)\sin(k_z z)\sin( \omega_p t)$$
# with
# $$ h^2 = k_x^2 + k_y^2 + k_z^2$$
# $$ k_x = \frac{m\pi}{L}$$
# $$ k_y = \frac{n\pi}{L}$$
# $$ k_z = \frac{p\pi}{L}$$

hi = [0.8, 0.8, 0.8]
lo = [-0.8, -0.8, -0.8]
m = 0
n = 1
p = 1
Lx = 1
Ly = 1
Lz = 1
h_2 = (m * pi / Lx) ** 2 + (n * pi / Ly) ** 2 + (p * pi / Lz) ** 2
theta = np.pi / 6

# Open the right plot file
filename = sys.argv[1]
ds = yt.load(filename)
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

t = ds.current_time.to_value()

rel_tol_err = 1e-2
my_grid = ds.index.grids[0]

By_sim = my_grid["raw", "By_fp"].squeeze().v
Bz_sim = my_grid["raw", "Bz_fp"].squeeze().v

ncells = np.array(np.shape(By_sim[:, :, :, 0]))
dx = (hi[0] - lo[0]) / ncells[0]
dy = (hi[1] - lo[1]) / ncells[1]
dz = (hi[2] - lo[2]) / ncells[2]

# Compute the analytic solution
Bx_th = np.zeros(ncells)
By_th = np.zeros(ncells)
Bz_th = np.zeros(ncells)
for i in range(ncells[0]):
    for j in range(ncells[1]):
        for k in range(ncells[2]):
            x0 = (i + 0.5) * dx + lo[0]
            y0 = j * dy + lo[1]
            z0 = (k + 0.5) * dz + lo[2]

            x = x0
            y = y0 * np.cos(-theta) - z0 * np.sin(-theta)
            z = y0 * np.sin(-theta) + z0 * np.cos(-theta)
            By = (
                -2
                / h_2
                * mu_0
                * (n * pi / Ly)
                * (p * pi / Lz)
                * (
                    np.cos(m * pi / Lx * (x - Lx / 2))
                    * np.sin(n * pi / Ly * (y - Ly / 2))
                    * np.cos(p * pi / Lz * (z - Lz / 2))
                    * np.cos(np.sqrt(2) * np.pi / Lx * c * t)
                )
            )

            Bz = mu_0 * (
                np.cos(m * pi / Lx * (x - Lx / 2))
                * np.cos(n * pi / Ly * (y - Ly / 2))
                * np.sin(p * pi / Lz * (z - Lz / 2))
                * np.cos(np.sqrt(2) * np.pi / Lx * c * t)
            )

            By_th[i, j, k] = (By * np.cos(theta) - Bz * np.sin(theta)) * (
                By_sim[i, j, k, 0] != 0
            )

            x0 = (i + 0.5) * dx + lo[0]
            y0 = (j + 0.5) * dy + lo[1]
            z0 = k * dz + lo[2]

            x = x0
            y = y0 * np.cos(-theta) - z0 * np.sin(-theta)
            z = y0 * np.sin(-theta) + z0 * np.cos(-theta)

            By = (
                -2
                / h_2
                * mu_0
                * (n * pi / Ly)
                * (p * pi / Lz)
                * (
                    np.cos(m * pi / Lx * (x - Lx / 2))
                    * np.sin(n * pi / Ly * (y - Ly / 2))
                    * np.cos(p * pi / Lz * (z - Lz / 2))
                    * np.cos(np.sqrt(2) * np.pi / Lx * c * t)
                )
            )

            Bz = mu_0 * (
                np.cos(m * pi / Lx * (x - Lx / 2))
                * np.cos(n * pi / Ly * (y - Ly / 2))
                * np.sin(p * pi / Lz * (z - Lz / 2))
                * np.cos(np.sqrt(2) * np.pi / Lx * c * t)
            )

            Bz_th[i, j, k] = (By * np.sin(theta) + Bz * np.cos(theta)) * (
                Bz_sim[i, j, k, 0] != 0
            )


# Compute relative l^2 error on By
rel_err_y = np.sqrt(
    np.sum(np.square(By_sim[:, :, :, 0] - By_th)) / np.sum(np.square(By_th))
)
assert rel_err_y < rel_tol_err
# Compute relative l^2 error on Bz
rel_err_z = np.sqrt(
    np.sum(np.square(Bz_sim[:, :, :, 0] - Bz_th)) / np.sum(np.square(Bz_th))
)
assert rel_err_z < rel_tol_err
