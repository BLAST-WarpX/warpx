#!/usr/bin/env python3
"""
RZ PSATD vacuum z-mode eigenmode precision test.

Initializes a purely axial (m=0) electromagnetic mode:
  Bz = B0 * sin(kz * z)

This is a z-direction-only mode that exercises the C2C FFT in the
z-direction without coupling to the radial Hankel transform. After
PSATD evolution in vacuum, the fields should match analytical predictions.

Uses parser-based B field initialization (Bz only; Br and Bt are
hardcoded to 0 in RZ parser mode). E fields default to zero.
"""

import argparse

import numpy as np
from pywarpx import picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument("--max_grid_size", type=int, default=None)
args, _ = parser.parse_known_args()

B0 = 1.0e-3       # T — magnetic field amplitude
c = constants.c
Lr = 1.0e-6        # radial extent
Lz = 1.0e-6        # axial extent
nr = 32
nz = 64
dr = Lr / nr
dz = Lz / nz
dt = 0.5 * dz / c
n_modes = 2

max_steps = 40

mgs = args.max_grid_size if args.max_grid_size else nz
grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    lower_bound=[0.0, 0.0],
    upper_bound=[Lr, Lz],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["none", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    n_azimuthal_modes=n_modes,
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
)

# B field initialization via parser.
# In RZ, the parser hardcodes Br=Bt=0 and only Bz is user-specified.
# Bz = B0 * sin(kz * z), E fields default to 0.
field_init = picmi.AnalyticInitialField(
    Bx_expression="0",
    By_expression="0",
    Bz_expression=f"{B0} * sin(2 * 3.141592653589793 * z / {Lz})",
)
sim.add_applied_field(field_init)

diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=max_steps,
    data_list=["Er", "Et", "Ez", "Br", "Bt", "Bz"],
    write_dir="diags",
)
sim.add_diagnostic(diag)

sim.step(max_steps)
