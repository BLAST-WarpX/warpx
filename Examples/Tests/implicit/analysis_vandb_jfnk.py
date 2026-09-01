#!/usr/bin/env python3

# Copyright 2024 Justin Angus
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from the scripts
# `inputs_test_2d_theta_implicit_jfnk_vandb` and
# `inputs_test_3d_theta_implicit_jfnk_vandb`.
# These simulate a periodic uniform plasma using the theta-implicit solver
# with shape factor 2. The dimensionality and the relevant algorithm options
# (deposition type, use of the mass matrices for the Jacobian) are read from
# the `warpx_used_inputs` file.
import sys

import numpy as np
import yt
from scipy.constants import e, epsilon_0

sys.path.append("../../../Tools/Parser/")
from input_file_parser import parse_input_file

input_dict = parse_input_file("./warpx_used_inputs")
dims = input_dict["geometry.dims"][0]
current_deposition = input_dict["algo.current_deposition"][0].strip('"')
use_mass_matrices_jacobian = input_dict.get(
    "implicit_evolve.use_mass_matrices_jacobian", ["false"]
)[0] in ("true", "1")

print(f"dimensionality: {dims}")
print(f"current deposition: {current_deposition}")
print(f"mass matrices used for the Jacobian: {use_mass_matrices_jacobian}")

field_energy = np.loadtxt("diags/reducedfiles/field_energy.txt", skiprows=1)
particle_energy = np.loadtxt("diags/reducedfiles/particle_energy.txt", skiprows=1)

total_energy = field_energy[:, 2] + particle_energy[:, 2]

delta_E = (total_energy - total_energy[0]) / total_energy[0]
max_delta_E = np.abs(delta_E).max()

# This case should have near machine precision conservation of energy
tolerance_rel_energy = 2.0e-14

print(f"max change in energy: {max_delta_E}")
print(f"tolerance: {tolerance_rel_energy}")

assert max_delta_E < tolerance_rel_energy

if current_deposition == "villasenor":
    # check for machine precision conservation of charge density
    tolerance_rel_charge = 2.0e-15
    n0 = 1.0e30

    pltdir = sys.argv[1]
    ds = yt.load(pltdir)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )

    divE = data["boxlib", "divE"].value
    rho = data["boxlib", "rho"].value

    # compute local error in Gauss's law
    drho = (rho - epsilon_0 * divE) / e / n0

    # compute RMS of the error on the grid
    drho2_avg = (drho**2).sum() / drho.size
    drho_rms = np.sqrt(drho2_avg)

    print(f"rms error in charge conservation: {drho_rms}")
    print(f"tolerance: {tolerance_rel_charge}")

    assert drho_rms < tolerance_rel_charge

if use_mass_matrices_jacobian:
    newton_solver = np.loadtxt("diags/newton_solver.txt", skiprows=1)
    num_steps = newton_solver[-1, 0]
    total_newton_iters = newton_solver[-1, 3]
    total_gmres_iters = newton_solver[-1, 7]

    # check that the number of gmres iterations per newton iteration is below
    # tolerance; this is sensitive to the quality of the mass-matrices-based
    # preconditioner
    gmres_iters_tol = 5.0
    print(f"gmres iters per newton: {total_gmres_iters / total_newton_iters}")
    print(f"gmres iters tolerance: {gmres_iters_tol}")
    assert total_gmres_iters / total_newton_iters <= gmres_iters_tol

    # check that the number of newton iterations per step is below tolerance;
    # this is sensitive to the quality of the mass-matrices-based Jacobian
    newton_iters_tol = 10.0
    print(f"newton iters per time step: {total_newton_iters / num_steps}")
    print(f"newton iters tolerance: {newton_iters_tol}")
    assert total_newton_iters / num_steps <= newton_iters_tol
