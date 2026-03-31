#!/usr/bin/env python3
"""
RZ PSATD Hankel radial-mode precision test.

Initializes a smooth radial Bz profile (Gaussian) that is uniform in z.
This exercises the Hankel transform forward/backward round-trip and
validates:
  - BLAS++ gemm (used for Hankel matrix multiply)
  - LAPACK++ getrf/getri (used for matrix inversion in HankelTransform ctor)
  - ILP64 correctness (wrong integer width crashes in lapack::getrf)

Uses parser-based B field initialization (Bz only; Br and Bt are
hardcoded to 0 in RZ parser mode). E fields default to zero.

After PSATD evolution in vacuum, the fields should evolve consistently
and preserve amplitude/shape to high precision.
"""

import argparse

from pywarpx import picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument("--max_grid_size", type=int, default=None)
args, _ = parser.parse_known_args()

B0 = 1.0e-3
c = constants.c
Lr = 1.0e-6
Lz = 1.0e-6
nr = 64
nz = 32             # must exceed guard cell count (default nox=16)
dr = Lr / nr
dz = Lz / nz
dt = 0.3 * min(dr, dz) / c
n_modes = 2

max_steps = 10

# Gaussian width: sigma = Lr/3 gives a profile that decays smoothly to
# near-zero at the radial boundary, avoiding boundary artifacts.
sigma = Lr / 3.0

mgs = args.max_grid_size if args.max_grid_size else max(nr, nz)
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

# Bz(r) = B0 * exp(-r^2 / (2*sigma^2)), uniform in z.
# In RZ parser mode, "x" is the radial coordinate r.
# Br and Bt are hardcoded to 0 by the RZ parser; E fields default to 0.
field_init = picmi.AnalyticInitialField(
    Bx_expression="0",
    By_expression="0",
    Bz_expression=f"{B0} * exp(-x*x / (2.0 * {sigma}**2))",
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
