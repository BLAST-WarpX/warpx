#!/usr/bin/env python3
"""Send an axisymmetric vacuum pulse across a full-length radial Yee join."""

import argparse
from pathlib import Path

import numpy as np

from pywarpx import amr, picmi, warpx

parser = argparse.ArgumentParser()
parser.add_argument("--direction", choices=("in", "out"), required=True)
parser.add_argument("--polarization", choices=("bz", "bt"), required=True)
parser.add_argument("--restart")
parser.add_argument("--reference")
parser.add_argument("--ordinary-restart", action="store_true")
parser.add_argument("--fixed-vacuum", action="store_true")
parser.add_argument("--particle-at-interface", action="store_true")
args = parser.parse_args()
c = picmi.constants.c
radius, dz, nz = 0.12, 0.0025, 8
nr = 192
dr = radius / (2 * nr)
dt = 0.4 * dr / c
steps = 288
outward = args.direction == "out"
te = args.polarization == "bz"
grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    lower_bound=[0, -nz * dz / 2],
    upper_bound=[radius, nz * dz / 2],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["open", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    n_azimuthal_modes=1,
    refined_regions=[[1, [0, -nz * dz / 2], [radius / 2, nz * dz / 2], [2, 1]]],
    warpx_max_grid_size=512,
    warpx_blocking_factor=4,
)
sim = picmi.Simulation(
    solver=picmi.ElectromagneticSolver(
        grid=grid,
        method="Yee",
        warpx_conforming_radial_interface=not args.ordinary_restart,
        warpx_conforming_radial_fixed_shape=args.fixed_vacuum,
    ),
    max_steps=steps,
    time_step_size=dt,
    warpx_use_filter=0,
    verbose=0,
    warpx_amr_restart=args.restart,
)
if args.particle_at_interface:
    sim.add_species(
        picmi.Species(
            name="electrons",
            particle_type="electron",
            initial_distribution=picmi.ParticleListDistribution(
                x=radius / 2, y=0.0, z=0.0, ux=0.0, uy=0.0, uz=0.0, weight=1.0
            ),
        ),
        layout=None,
    )
sim.add_diagnostic(
    picmi.Checkpoint(
        name="chk", period=steps // 2, write_dir="diags", warpx_file_prefix="chk"
    )
)
sim.initialize_inputs()
amr.n_error_buf = 0
warpx.do_initial_div_cleaning = 0
warpx.pml_ncell = 4
warpx.pml_delta = 4
sim.initialize_warpx()
components = {"B": "z" if te else "theta", "E": "theta" if te else "z"}


def initial(r, kind):
    center = 0.04 if outward else 0.08
    sign = (1 if outward else -1) * (1 if te else -1)
    return (1 if kind == "B" else sign * c) * np.exp(-0.5 * ((r - center) / 0.004) ** 2)


def field(kind, level, patch):
    mf = sim.fields.get(f"{kind}field_{patch}", dir=components[kind], level=level)
    values = np.asarray(mf[...]).copy()
    ba = mf.box_array()
    spacing = dr if (level, patch) == (1, "fp") else 2 * dr
    r = (
        np.arange(values.shape[0])
        + ba.minimal_box().small_end[0]
        + 0.5 * (1 - ba.ix_type().node_centered(0))
    ) * spacing
    return mf, values, r


if not args.restart:
    for level, patch in ((0, "fp"), (1, "fp"), (1, "cp")):
        for kind in components:
            mf, values, r = field(kind, level, patch)
            values[...] = initial(r, kind).reshape((-1,) + (1,) * (values.ndim - 1))
            mf[...] = values
sim.step(steps // 2 if args.restart else steps)
if args.fixed_vacuum:
    assert Path(f"diags/chk{steps:06d}/RadialFixedShape").is_file()

# Independent radial Yee solve on a globally fine mesh. The sampled region and
# flight time keep the pulse clear of the radial conducting reference boundary.
r_e = dr * np.arange(2 * nr + 1)
r_b = (r_e[:-1] + r_e[1:]) / 2
e = initial(r_e, "E")
b = initial(r_b, "B")


def advance_b():
    if te:
        b[:] -= 0.5 * dt * np.diff(r_e * e) / (r_b * dr)
    else:
        b[:] += 0.5 * dt * np.diff(e) / dr


for _ in range(steps):
    advance_b()
    if te:
        e[1:-1] -= c**2 * dt * np.diff(b) / dr
        e[0] = 0.0
    else:
        e[1:-1] += c**2 * dt * np.diff(r_b * b) / (r_e[1:-1] * dr)
        e[0] += c**2 * dt * 4 * b[0] / dr
    advance_b()

native = {}
for kind, r, expected in (("E", r_e, e), ("B", r_b, b)):
    _, fine, rf = field(kind, 1, "fp")
    _, parent, rp = field(kind, 0, "fp")
    outside = rp > radius / 2 + 1.0e-12
    x = np.r_[rf, rp[outside]]
    values = np.r_[np.squeeze(fine)[:, nz // 2], np.squeeze(parent)[outside, nz // 2]]
    assert np.all(np.diff(x) > 0)
    mask = (r > 0.012) & (r < 0.107)
    residual = np.interp(r[mask], x, values) - expected[mask]
    error = np.sqrt(
        np.sum(r[mask] * residual**2) / np.sum(r[mask] * expected[mask] ** 2)
    )
    print(f"{kind} radial pulse relative error {error:.6g}")
    assert error < 0.007
    native[kind] = values
if args.reference:
    with np.load(args.reference) as ref:
        for kind, values in native.items():
            np.testing.assert_allclose(
                values, ref[kind], atol=1e-13 * np.max(abs(values)), rtol=1e-13
            )
np.savez("radial_fields.npz", **native)
