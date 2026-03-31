#!/usr/bin/env python3
"""
2D PSATD vacuum eigenmode precision test.

Initializes a forward-propagating electromagnetic plane wave as a single
eigenmode of the discrete Fourier transform. In vacuum (no particles, no
currents), the PSATD update is an exact phase rotation — the fields should
match the analytical solution to near-machine precision after many steps.

Mode: Ey = E0 * sin(2*pi*x/Lx), Bz = +(E0/c) * sin(2*pi*x/Lx)
This satisfies div(E)=0 and is a +x-propagating eigenmode.
"""

import argparse
import sys

from pywarpx import picmi

constants = picmi.constants

# Parse optional max_grid_size override for multi-box tests
parser = argparse.ArgumentParser()
parser.add_argument("--max_grid_size", type=int, default=None)
parser.add_argument("--ncells", type=int, default=32,
                    help="Grid cells per dimension (must exceed guard cell count)")
args, _ = parser.parse_known_args()

# --- Physical and numerical parameters ---
E0 = 1.0e5       # V/m — field amplitude
c = constants.c
Lx = 1.0e-6      # domain size in x (m)
Lz = 1.0e-6      # domain size in z (m)
nx = args.ncells
nz = args.ncells
dx = Lx / nx
dz = Lz / nz
dt = 0.5 * min(dx, dz) / c  # CFL < 1

max_steps = 40

# Wave parameters for the initialized mode
kx = 2.0 * 3.141592653589793 / Lx  # single mode in x
omega = c * kx                       # dispersion relation: omega = c|k|

# --- Grid ---
mgs = args.max_grid_size if args.max_grid_size else max(nx, nz)
grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, nz],
    lower_bound=[0.0, 0.0],
    upper_bound=[Lx, Lz],
    lower_boundary_conditions=["periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic"],
    warpx_max_grid_size=mgs,
)

# --- Solver ---
solver = picmi.ElectromagneticSolver(
    grid=grid,
    method="PSATD",
    cfl=0.99,  # unused when dt is set explicitly, but required
)

# --- Simulation ---
sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=max_steps,
    verbose=0,
    warpx_grid_type="collocated",
    warpx_use_filter=False,
)

# --- Field initialization ---
# Forward-propagating eigenmode in +x direction:
# Ey = E0 * sin(kx * x)
# Bz = +(E0/c) * sin(kx * x)
field_init = picmi.AnalyticInitialField(
    Ex_expression="0",
    Ey_expression=f"{E0} * sin(2 * 3.141592653589793 * x / {Lx})",
    Ez_expression="0",
    Bx_expression="0",
    By_expression="0",
    Bz_expression=f"{E0/c} * sin(2 * 3.141592653589793 * x / {Lx})",
)
sim.add_applied_field(field_init)

# --- Diagnostics ---
diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=max_steps,
    data_list=["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
    write_dir="diags",
)
sim.add_diagnostic(diag)

# --- Run ---
sim.step(max_steps)
