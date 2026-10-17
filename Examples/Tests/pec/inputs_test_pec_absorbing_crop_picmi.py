#!/usr/bin/env python3
#
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
"""Discrete Gauss's law next to an absorbing PEC wall with trajectory cropping.

Charged rings (RZ) or sheets (XZ) are born in the interior moving towards the upper
x (or r) boundary, which is a PEC field boundary with absorbing particles. With
``particles.crop_on_PEC_boundary = 1`` the explicit Esirkepov deposition stops each
absorbed particle's current at the wall, so the image fold of the PEC boundary stays
charge consistent and eps0*div(E) - rho vanishes on every interior node away from
the birth location (birth adds charge without a current, by construction).

Usage: inputs_test_pec_absorbing_crop_picmi.py --dim {rz,xz} --shape {1,2,3}
"""

import argparse
import gc

import numpy as np

from pywarpx import callbacks, picmi
from pywarpx import particles as particles_bucket

parser = argparse.ArgumentParser()
parser.add_argument("--dim", choices=["rz", "xz"], default="rz")
parser.add_argument("--shape", type=int, default=1)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument(
    "--no-crop", action="store_true", help="diagnostic: leave cropping off, report only"
)
args = parser.parse_args()

EPS0 = picmi.constants.ep0
Q_E = picmi.constants.q_e
NX, NZ = 32, 8
X_MAX, LENGTH = 4.0e-2, 4.0e-2
X_BIRTH = 1.53e-2
WEIGHT = 1.0e6
VELOCITY = 2.0e7
TOLERANCE = 1.0e-10  # interior Gauss residual relative to the collected charge

lower_bc = ["none" if args.dim == "rz" else "dirichlet", "periodic"]
lower_bc_particles = ["none" if args.dim == "rz" else "absorbing", "periodic"]
grid_args = dict(
    number_of_cells=[NX, NZ],
    lower_bound=[0.0, 0.0],
    upper_bound=[X_MAX, LENGTH],
    lower_boundary_conditions=lower_bc,
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=lower_bc_particles,
    upper_boundary_conditions_particles=["absorbing", "periodic"],
)
if args.dim == "rz":
    grid = picmi.CylindricalGrid(n_azimuthal_modes=1, **grid_args)
else:
    grid = picmi.Cartesian2DGrid(**grid_args)

solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)
sim = picmi.Simulation(
    solver=solver,
    max_steps=args.steps,
    particle_shape=args.shape,
    warpx_current_deposition_algo="esirkepov",
    warpx_use_filter=False,
    verbose=0,
)
sim.add_species(
    picmi.Species(
        name="electrons", particle_type="electron", initial_distribution=None
    ),
    layout=None,
)
# No PICMI keyword exists for this option; set it in the input bucket.
particles_bucket.crop_on_PEC_boundary = 0 if args.no_crop else 1
sim.initialize_inputs()
sim.initialize_warpx()

warpx = sim.extension.warpx
species = sim.particles.get("electrons")
dx, dz = X_MAX / NX, LENGTH / NZ
x_nodes = np.arange(NX + 1) * dx
x_half = (np.arange(NX) + 0.5) * dx
z_birth = (np.arange(NZ) + 0.5) * dz
X_DIR = "r" if args.dim == "rz" else "x"
if args.dim == "rz":
    node_volume = 2.0 * np.pi * x_nodes * dx * dz
    metric_nodes, metric_half = x_nodes, x_half
else:
    node_volume = np.full(NX + 1, dx * dz)
    metric_nodes, metric_half = np.ones(NX + 1), np.ones(NX)
inject_steps = args.steps // 4


def deposited_rho():
    """Freshly deposited charge density (RZ volume scaling and guard sum included)."""
    return species.get_charge_density(0, False)[...][:, :NZ]


def divergence():
    """Nodal Yee divergence on interior x (r) nodes, periodic z.

    Computed here because WarpX's nodal ComputeDivE has no Python binding and the
    divE diagnostic is cell-centered."""
    ex = sim.fields.get("Efield_fp", dir=X_DIR, level=0)[...][:, :NZ]
    ez = sim.fields.get("Efield_fp", dir="z", level=0)[...][:, :NZ]
    div = np.zeros((NX + 1, NZ))
    for i in range(1, NX):
        div[i] = (metric_half[i] * ex[i] - metric_half[i - 1] * ex[i - 1]) / (
            metric_nodes[i] * dx
        )
    div += (ez - np.roll(ez, 1, axis=1)) / dz
    return div


def inject():
    if warpx.getistep(0) < inject_steps:
        species.add_particles(
            x=np.full(NZ, X_BIRTH),
            y=np.zeros(NZ),
            z=z_birth,
            ux=np.full(NZ, VELOCITY),
            uy=np.zeros(NZ),
            uz=np.zeros(NZ),
            w=np.full(NZ, WEIGHT),
        )


callbacks.installbeforestep(inject)
sim.step(args.steps)

residual = ((EPS0 * divergence() - deposited_rho()) * node_volume[:, None]).sum(axis=1)
away_from_birth = np.abs(x_nodes - X_BIRTH) > 2.5 * dx
interior = np.zeros(NX + 1, dtype=bool)
interior[1:NX] = True
check = interior & away_from_birth
live = species.total_number_of_particles(True, False)
collected = Q_E * WEIGHT * (NZ * inject_steps - live)
worst = np.max(np.abs(residual[check])) / collected
print(
    f"dim={args.dim} shape={args.shape}: collected {collected:.4e} C, "
    f"worst interior Gauss residual / collected = {worst:.3e}"
)
assert collected > 0.5 * Q_E * WEIGHT * NZ * inject_steps, "too few particles absorbed"
if not args.no_crop:
    assert worst < TOLERANCE, f"Gauss residual {worst:.3e} exceeds {TOLERANCE:.1e}"

# Python-owned AMReX objects must be released before AMReX finalizes.
callbacks.uninstallcallback("beforestep", inject)
species = None
gc.collect()
sim.finalize()
