#!/usr/bin/env python3

# Copyright 2026 Debojyoti Ghosh
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# Analysis script for 2D planar pinch tests exercising different
# linear solvers and preconditioners (Jacobi PC, Chebyshev PC,
# weighted Jacobi linear solver, Chebyshev linear solver).
# Checks solver convergence and charge conservation.
#
# Usage: analysis_planar_pinch_solvers.py <plotdir>
#            <newton_iters_tol> <linsol_iters_tol>

import sys

import numpy as np
import yt
from scipy.constants import e, epsilon_0

pltdir = sys.argv[1]
newton_iters_tol = int(sys.argv[2])
linsol_iters_tol = int(sys.argv[3])

newton_solver = np.loadtxt("diags/reduced_files/newton_solver.txt", skiprows=1)
if newton_solver.ndim == 1:
    newton_solver = newton_solver.reshape(1, -1)
num_steps = newton_solver[-1, 0]
total_newton_iters = newton_solver[-1, 3]
total_linsol_iters = newton_solver[-1, 7]

# check that Newton converged (relative norm < 1e-4)
newton_rel_norm = newton_solver[-1, 5]
print(f"newton relative norm at last step: {newton_rel_norm}")
assert newton_rel_norm < 1.0e-4

# check that Newton iterations per step are reasonable
newton_iters_per_step = total_newton_iters / num_steps
print(f"newton iters per time step: {newton_iters_per_step}")
print(f"newton iters tolerance: {newton_iters_tol}")
assert newton_iters_per_step < newton_iters_tol

# check that linear solver iterations per Newton iteration are reasonable
linsol_iters_per_newton = total_linsol_iters / total_newton_iters
print(f"linear solver iters per newton: {linsol_iters_per_newton}")
print(f"linear solver iters tolerance: {linsol_iters_tol}")
assert linsol_iters_per_newton < linsol_iters_tol

# check for conservation of charge density (Gauss's law)
n0 = 1.0e23

ds = yt.load(pltdir)
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

divE = data["boxlib", "divE"].value
rho = data["boxlib", "rho"].value

drho = (rho - epsilon_0 * divE) / e / n0

# excluding the upper boundary where the insulator is located
drho_trimmed = drho[:-1, ...]
Ng = drho_trimmed.size
drho2_avg = (drho_trimmed**2).sum() / Ng
drho_rms = np.sqrt(drho2_avg)
tolerance_rel_charge = 5.0e-6
print(f"rms error in charge conservation: {drho_rms}")
print(f"tolerance: {tolerance_rel_charge}")
assert drho_rms < tolerance_rel_charge
