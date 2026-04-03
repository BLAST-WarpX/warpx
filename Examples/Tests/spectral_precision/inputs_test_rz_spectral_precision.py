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

parser = argparse.ArgumentParser(
    description="RZ PSATD vacuum z-mode eigenmode precision test. "
    "Initializes a purely axial Bz mode and evolves with PSATD. "
    "Grid size and decomposition can be adjusted for multi-rank runs. "
    "Domain decomposition is along z only (r is not split). "
    "Requires pywarpx (with WarpX_FFT=ON) to be installed.")
parser.add_argument("--nr", type=int, default=32,
                    help="Radial cells (default: 32). "
                    "Must exceed the PSATD guard cell count (default nox=16).")
parser.add_argument("--nz", type=int, default=64,
                    help="Axial cells (default: 64). "
                    "Must exceed the PSATD guard cell count (default nox=16). "
                    "Increase for more ranks, e.g. --nz 128 for 4 boxes.")
parser.add_argument("--max_grid_size", type=int, default=None,
                    help="Maximum box size for domain decomposition (default: nz, "
                    "i.e. single box). Set smaller than nz to create multiple "
                    "boxes, e.g. --nz 128 --max_grid_size 32 gives 4 boxes.")
args, _ = parser.parse_known_args()

import numpy as np
from pywarpx import picmi

constants = picmi.constants

B0 = 1.0e-3       # T — magnetic field amplitude
c = constants.c
Lr = 1.0e-6        # radial extent
Lz = 1.0e-6        # axial extent
nr = args.nr
nz = args.nz
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
