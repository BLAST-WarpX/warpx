#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Discrete Gauss law next to an embedded boundary while particles are absorbed.

RZ, Yee, Esirkepov, periodic z. An EB rod on the axis absorbs electron rings that
are injected outside it and move inward. Near the EB, Esirkepov deposits the
current with a reduced (order 1) particle shape; the charge density must use the
same shape there, otherwise eps0 div(E) - rho is nonzero on regular nodes next
to the EB while particles are present (shapes > 1).

div(E) is evaluated here from Efield_fp with the cylindrical Yee divergence and
rho is a fresh deposit (get_charge_density), so the check is independent of the
diagnostic interpolation (the divE diagnostic is cell-centered, and WarpX's
nodal ComputeDivE has no Python binding). Nodes near the injection radius are
excluded: ring creation adds charge without a current there by construction.
"""

import argparse
import gc

import numpy as np

from pywarpx import callbacks, picmi

parser = argparse.ArgumentParser()
parser.add_argument("--shape", type=int, default=2)
args = parser.parse_args()

NR, NZ = 48, 8
R_ROD, R_WALL, LENGTH = 6.2e-3, 6.0e-2, 4.0e-2
R_BIRTH, WEIGHT, STEPS = 1.53e-2, 1.0e6, 1500
EPS0 = picmi.constants.ep0
Q_E = picmi.constants.q_e
TOLERANCE = 1.0e-8  # relative to the injected charge

grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    n_azimuthal_modes=1,
    lower_bound=[0.0, 0.0],
    upper_bound=[R_WALL, LENGTH],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    warpx_blocking_factor=8,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)
embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function=f"{R_ROD}*{R_ROD}-(x*x+y*y)", potential=0.0
)
sim = picmi.Simulation(
    solver=solver,
    max_steps=STEPS,
    particle_shape=args.shape,
    warpx_embedded_boundary=embedded_boundary,
    warpx_use_filter=False,
    warpx_current_deposition_algo="esirkepov",
    verbose=0,
)
electrons = picmi.Species(
    name="electrons", particle_type="electron", initial_distribution=None
)
sim.add_species(electrons, layout=None)
sim.initialize_inputs()
sim.initialize_warpx()

warpx = sim.extension.warpx
particles = sim.particles.get("electrons")
dr, dz = R_WALL / NR, LENGTH / NZ
r_node = np.arange(NR + 1) * dr
r_half = (np.arange(NR) + 0.5) * dr
ring_volume = 2.0 * np.pi * r_node * dr * dz
z_birth = (np.arange(NZ) + 0.5) * dz
inject_steps = STEPS // 3
# regular nodes between the EB surface region and the injection region
checked = (r_node > R_ROD + dr) & (r_node < R_BIRTH - 2.5 * dr)
history = []


def deposited_rho():
    """Freshly deposited charge density (RZ volume scaling and guard sum included)."""
    return particles.get_charge_density(0, False)[...][:, :NZ]


def divergence():
    er = sim.fields.get("Efield_fp", dir="r", level=0)[...][:, :NZ]
    ez = sim.fields.get("Efield_fp", dir="z", level=0)[...][:, :NZ]
    div = np.zeros((NR + 1, NZ))
    for i in range(1, NR):
        div[i] = (r_half[i] * er[i] - r_half[i - 1] * er[i - 1]) / (r_node[i] * dr)
    div += (ez - np.roll(ez, 1, axis=1)) / dz
    return div


def inject():
    if warpx.getistep(0) < inject_steps:
        particles.add_particles(
            x=np.full(NZ, R_BIRTH),
            y=np.zeros(NZ),
            z=z_birth,
            ux=np.full(NZ, -1.0e7),
            uy=np.zeros(NZ),
            uz=np.zeros(NZ),
            w=np.full(NZ, WEIGHT),
            unique_particles=False,
        )


def record():
    if warpx.getistep(0) % 50:
        return
    residual = ((EPS0 * divergence() - deposited_rho()) * ring_volume[:, None]).sum(
        axis=1
    )
    history.append(float(np.max(np.abs(residual[checked]))))


callbacks.installbeforestep(inject)
callbacks.installafterstep(record)
sim.step(STEPS)

injected_charge = Q_E * WEIGHT * NZ * inject_steps
error = max(history) / injected_charge
print(
    f"shape {args.shape}: max |eps0 divE - rho| near the EB / injected charge = {error:.3e}"
)

callbacks.uninstallcallback("beforestep", inject)
callbacks.uninstallcallback("afterstep", record)
particles = None
gc.collect()
sim.finalize()

assert error < TOLERANCE, f"Gauss-law error {error:.3e} exceeds {TOLERANCE:.1e}"
