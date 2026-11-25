#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
Test that the magnetostatic solver puts B where the current is.

A ring of protons, centered on the z axis, moves azimuthally (a current loop).
The domain, the grid and the particle loading are symmetric under a rotation by
180 degrees about the z axis, (x, y) -> (-x, -y). Under this rotation, Bz is
unchanged and (Bx, By) change sign, so the magnetostatic B field must have the
same symmetry, to round-off and solver tolerance.

A half-cell error in the interpolation of the derivatives of the vector
potential A onto the B grid shifts dAy/dx along x and dAx/dy along y, which
breaks this symmetry for Bz (by ~30% of max|B| in this setup). The test checks
  1. the symmetry of Bx, By, Bz under the 180-degree rotation, and
  2. the centroid of the Bz depression/peak inside the ring (in cells).
"""

import numpy as np
import scipy.constants as con

from pywarpx import picmi

##########################
# numerics parameters
##########################

dt = 1e-12
max_steps = 1

nx = ny = nz = 64
xmin, xmax = -0.25, 0.25
ymin, ymax = -0.25, 0.25
zmin, zmax = -0.25, 0.25

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    warpx_max_grid_size=64,
)

solver = picmi.ElectrostaticSolver(
    grid=grid,
    method="Multigrid",
    required_precision=1e-10,
    warpx_magnetostatic=True,
    warpx_self_fields_verbosity=0,
)

##########################
# current ring
##########################

ring_radius = 0.1  # m
ring_halfwidth = 0.02  # m (radial)
ring_halfheight = 0.02  # m (axial)
n0 = 1.0e14  # m^-3
u0 = 1.0e6  # m/s, azimuthal

ring = picmi.AnalyticDistribution(
    density_expression="(abs(sqrt(x*x+y*y)-a)<w)*(abs(z)<h)*n0",
    momentum_expressions=[
        "-u0*y/sqrt(x*x+y*y+1.e-30)",
        "u0*x/sqrt(x*x+y*y+1.e-30)",
        None,
    ],
    a=ring_radius,
    w=ring_halfwidth,
    h=ring_halfheight,
    n0=n0,
    u0=u0,
)
protons = picmi.Species(
    particle_type="proton", name="protons", initial_distribution=ring
)
layout = picmi.GriddedLayout(n_macroparticle_per_cell=[2, 2, 2], grid=grid)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=max_steps,
    warpx_current_deposition_algo="direct",
    warpx_use_filter=False,
    warpx_serialize_initial_conditions=True,
    warpx_do_dynamic_scheduling=False,
)
sim.add_species(protons, layout=layout)

sim.step(max_steps)

##########################
# Testing
##########################


def get_B(comp):
    """Global numpy array of the (staggered) B component on level 0."""
    return np.asarray(sim.fields.get("Bfield_fp", dir=comp, level=0)[...])


def coords(n_points, lo, hi, n_cells):
    """Cell-centered or nodal coordinates, from the number of points."""
    d = (hi - lo) / n_cells
    offset = 0.5 if n_points == n_cells else 0.0
    return lo + (np.arange(n_points) + offset) * d


B = {c: get_B(c) for c in "xyz"}
Bmax = max(np.abs(b).max() for b in B.values())
assert Bmax > 0.0, "the magnetostatic solver returned B = 0"

# 1. symmetry under (x, y) -> (-x, -y): Bz even, Bx and By odd
asym = {}
for comp, sign in (("x", -1.0), ("y", -1.0), ("z", 1.0)):
    b = B[comp]
    asym[comp] = np.abs(b - sign * b[::-1, ::-1, :]).max() / Bmax
    print(f"B{comp}: max |B - R(B)| / max|B| = {asym[comp]:.3e}")

# 2. centroid of Bz inside the ring, in the plane closest to z = 0
bz = B["z"]
x = coords(bz.shape[0], xmin, xmax, nx)
y = coords(bz.shape[1], ymin, ymax, ny)
z = coords(bz.shape[2], zmin, zmax, nz)
plane = bz[:, :, np.argmin(np.abs(z))]
X, Y = np.meshgrid(x, y, indexing="ij")
inside = X**2 + Y**2 < (0.5 * ring_radius) ** 2
w = plane[inside]
dx = (xmax - xmin) / nx
xc = np.sum(w * X[inside]) / np.sum(w) / dx
yc = np.sum(w * Y[inside]) / np.sum(w) / dx
print(f"Bz centroid inside the ring: ({xc:+.3e}, {yc:+.3e}) cells")

# Analytical scale check: field at the center of a thin loop, mu0 I / (2 a)
current = con.e * n0 * u0 * (2 * ring_halfwidth) * (2 * ring_halfheight)
bz_center_thin_loop = con.mu_0 * current / (2 * ring_radius)
bz_center = plane[np.unravel_index(np.argmin(X**2 + Y**2), X.shape)]
print(
    f"Bz near the center: {bz_center:.4e} T (thin-loop estimate {bz_center_thin_loop:.4e} T)"
)

tol_sym = 1.0e-6
tol_centroid = 1.0e-3  # cells
assert max(asym.values()) < tol_sym, f"B breaks the 180-degree symmetry: {asym}"
assert abs(xc) < tol_centroid and abs(yc) < tol_centroid, (
    f"Bz is not centered on the current ring: ({xc}, {yc}) cells"
)
assert 0.5 < bz_center / bz_center_thin_loop < 1.5, (
    "Bz at the center is far from mu0 I / (2a)"
)
print("PASS")
