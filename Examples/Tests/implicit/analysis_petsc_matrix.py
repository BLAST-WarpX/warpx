#!/usr/bin/env python3

# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses simulation results from the following input files:
# `inputs_test_2d_curl_curl_petsc_pc`,
# `inputs_test_rcylinder_curl_curl_petsc_pc`, and
# `inputs_test_rz_curl_curl_petsc_pc`.
# These tests simulate a time-varing EM wave injected from an insulator boundary
# using the theta-implicit solver with PETSc's LU preconditer.
# Since LU is an exact solver, if the preconditioner matrix is constructed
# correctly, then here should be 1 Newton and 1 GMRES iteration per time step.

import numpy as np

newton_solver = np.loadtxt("diags/reduced_files/newton_solver.txt", skiprows=1)
num_steps = newton_solver[-1, 0]
total_newton_iters = newton_solver[-1, 3]
total_gmres_iters = newton_solver[-1, 7]

# check that there is 1 Newton and 1 GMRES iteration per step
print(f"total steps: {num_steps}")
print(f"total gmres iters: {total_gmres_iters}")
print(f"total newton iters: {total_newton_iters}")
assert total_gmres_iters == num_steps
assert total_newton_iters == num_steps
