#!/usr/bin/env python3
"""
3D PSATD vacuum eigenmode precision test.

Initializes a forward-propagating electromagnetic plane wave:
  Ey = E0 * sin(kz * z)
  Bx = (E0/c) * sin(kz * z)   [+z-propagating mode]

In vacuum, the PSATD update is an exact phase rotation, so the fields
should match the analytical solution to near-machine precision.
"""

import argparse

from pywarpx import picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument("--max_grid_size", type=int, default=None)
args, _ = parser.parse_known_args()

E0 = 1.0e5
c = constants.c
L = 1.0e-6
n = 32  # 32^3 grid — must exceed default guard cell count (nox=16)
dx = L / n
dt = 0.3 * dx / c  # CFL for 3D: dt < dx/(c*sqrt(3))

max_steps = 40

mgs = args.max_grid_size if args.max_grid_size else n
grid = picmi.Cartesian3DGrid(
    number_of_cells=[n, n, n],
    lower_bound=[0.0, 0.0, 0.0],
    upper_bound=[L, L, L],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
    warpx_max_grid_size=mgs,
)

solver = picmi.ElectromagneticSolver(
    grid=grid,
    method="PSATD",
    cfl=0.99,
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=max_steps,
    verbose=0,
    warpx_grid_type="collocated",
    warpx_use_filter=False,
)

# +z-propagating mode (single mode along z only)
field_init = picmi.AnalyticInitialField(
    Ex_expression="0",
    Ey_expression=f"{E0} * sin(2 * 3.141592653589793 * z / {L})",
    Ez_expression="0",
    Bx_expression=f"{E0/c} * sin(2 * 3.141592653589793 * z / {L})",
    By_expression="0",
    Bz_expression="0",
)
sim.add_applied_field(field_init)

diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=max_steps,
    data_list=["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
    write_dir="diags",
)
sim.add_diagnostic(diag)

sim.step(max_steps)
