#!/usr/bin/env python3
#
# Copyright 2019-2021 David Bizzozero, David Grote
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
This script tests the expansion of a uniformly charged sphere of electrons.

An input file inputs_3d or inputs_rz is used: it defines a sphere of charge with
given initial conditions. The sphere of charge starts at rest and will expand
due to Coulomb interactions. The test will check if the sphere has expanded at
the correct speed and that the electric field is accurately modeled against a
known analytic solution. While the radius r(t) is not analytically known, its
inverse t(r) can be solved for exactly.
"""

import os
import re
import sys

import numpy as np
import yt
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c
from scipy.optimize import fsolve

yt.funcs.mylog.setLevel(0)

# Open plotfile specified in command line
test_name = os.path.split(os.getcwd())[1]
filename = sys.argv[1]
ds = yt.load(filename)
t_max = ds.current_time.item()  # time of simulation

# Parse test name and check if particle_shape = 4 is used
emass_10 = True if re.search("emass_10", test_name) else False

if emass_10:
    l2_tolerance = 0.096
    m_e = 10
else:
    l2_tolerance = 0.05
    m_e = 9.10938356e-31  # Electron mass in kg
ndims = np.count_nonzero(ds.domain_dimensions > 1)

if ndims == 2:
    xmin, zmin = [float(x) for x in ds.parameters.get("geometry.prob_lo").split()]
    xmax, zmax = [float(x) for x in ds.parameters.get("geometry.prob_hi").split()]
    nx, nz = [int(n) for n in ds.parameters["amr.n_cell"].split()]
    ymin, ymax = xmin, xmax
    ny = nx
else:
    xmin, ymin, zmin = [float(x) for x in ds.parameters.get("geometry.prob_lo").split()]
    xmax, ymax, zmax = [float(x) for x in ds.parameters.get("geometry.prob_hi").split()]
    nx, ny, nz = [int(n) for n in ds.parameters["amr.n_cell"].split()]

dx = (xmax - xmin) / nx
dy = (ymax - ymin) / ny
dz = (zmax - zmin) / nz

# Grid location of the axis
ix0 = round((0.0 - xmin) / dx)
iy0 = round((0.0 - ymin) / dy)
iz0 = round((0.0 - zmin) / dz)

# Constants
eps_0 = 8.8541878128e-12  # Vacuum Permittivity in C/(V*m)
q_e = -1.60217662e-19  # Electron charge in C
pi = np.pi  # Circular constant of the universe
r_0 = 0.1  # Initial radius of sphere
q_tot = -1e-15  # Total charge of sphere in C


# Define functions for exact forms of v(r), t(r), Er(r) with r as the radius of
# the sphere. The sphere starts with initial radius r_0 and this radius expands
# with zero initial velocity. Note: v(t) and r(t) are not known analytically but
# v(r) and t(r) can be solved analytically.
#
# The solution r(t) solves the ODE: r''(t) = a/(r(t)**2) with initial conditions
# r(0) = r_0, r'(0) = 0, and a = q_e*q_tot/(4*pi*eps_0*m_e)
#
# The E was calculated at the end of the last time step
def v_exact(r):
    return np.sqrt(q_e * q_tot / (2 * pi * m_e * eps_0) * (1 / r_0 - 1 / r))


def t_exact(r):
    return np.sqrt(r_0**3 * 2 * pi * m_e * eps_0 / (q_e * q_tot)) * (
        np.sqrt(r / r_0 - 1) * np.sqrt(r / r_0)
        + np.log(np.sqrt(r / r_0 - 1) + np.sqrt(r / r_0))
    )


def func(rho):
    return t_exact(rho) - t_max  # Objective function to find r(t_max)


r_end = fsolve(func, r_0)[0]  # Numerically solve for r(t_max)


def E_exact(r):
    return np.sign(r) * (
        q_tot / (4 * pi * eps_0 * r**2) * (abs(r) >= r_end)
        + q_tot * abs(r) / (4 * pi * eps_0 * r_end**3) * (abs(r) < r_end)
    )


# Load data pertaining to fields
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

# Extract the E field along the axes
# if ndims == 2:
if ds.parameters["geometry.dims"] == "RZ":
    Ex = data[("boxlib", "Er")].to_ndarray()
    Ex_axis = Ex[:, iz0, 0]
    Ey_axis = Ex_axis
    Ez = data[("boxlib", "Ez")].to_ndarray()
    Ez_axis = Ez[ix0, :, 0]
else:
    Ex = data[("mesh", "Ex")].to_ndarray()
    Ex_axis = Ex[:, iy0, iz0]
    Ey = data[("mesh", "Ey")].to_ndarray()
    Ey_axis = Ey[ix0, :, iz0]
    Ez = data[("mesh", "Ez")].to_ndarray()
    Ez_axis = Ez[ix0, iy0, :]


def calculate_error(E_axis, xmin, dx, nx):
    # Compute cell centers for grid
    x_cell_centers = np.linspace(xmin + dx / 2.0, xmax - dx / 2.0, nx)

    # Extract subgrid away from boundary (exact solution assumes infinite/open
    # domain but WarpX solution assumes perfect conducting walls)
    ix1 = round((xmin / 2.0 - xmin) / dx)
    ix2 = round((xmax / 2.0 - xmin) / dx)
    x_sub_grid = x_cell_centers[ix1:ix2]

    # Exact solution of field along Cartesian axes
    E_exact_grid = E_exact(x_sub_grid)

    # WarpX solution of field along Cartesian axes
    E_grid = E_axis[ix1:ix2]

    # Define approximate L2 norm error between exact and numerical solutions
    L2_error = np.sqrt(sum((E_exact_grid - E_grid) ** 2)) / np.sqrt(
        sum((E_exact_grid) ** 2)
    )

    return L2_error


L2_error_x = calculate_error(Ex_axis, xmin, dx, nx)
L2_error_y = calculate_error(Ey_axis, ymin, dy, ny)
L2_error_z = calculate_error(Ez_axis, zmin, dz, nz)
print("L2 error along x-axis = %s" % L2_error_x)
print("L2 error along y-axis = %s" % L2_error_y)
print("L2 error along z-axis = %s" % L2_error_z)

assert L2_error_x < l2_tolerance
assert L2_error_y < l2_tolerance
assert L2_error_z < l2_tolerance


# Check conservation of energy
def return_energies(iteration):
    ux, uy, uz, phi, m, q, w = ts.get_particle(
        ["ux", "uy", "uz", "phi", "mass", "charge", "w"], iteration=iteration
    )
    E_kinetic = (w * m * c**2 * (np.sqrt(1 + ux**2 + uy**2 + uz**2) - 1)).sum()
    E_potential = (
        0.5 * (w * q * phi).sum()
    )  # potential energy of particles in their own space-charge field: includes factor 1/2
    return E_kinetic, E_potential


ts = OpenPMDTimeSeries("./diags/diag2")
if "phi" in ts.avail_record_components["electron"]:
    # phi is only available when this script is run with the labframe poisson solver
    print("Checking conservation of energy")
    Ek_i, Ep_i = return_energies(0)
    Ek_f, Ep_f = return_energies(30)
    assert Ep_f < 0.7 * Ep_i  # Check that potential energy changes significantly
    assert abs((Ek_i + Ep_i) - (Ek_f + Ep_f)) < 0.003 * (
        Ek_i + Ep_i
    )  # Check conservation of energy
