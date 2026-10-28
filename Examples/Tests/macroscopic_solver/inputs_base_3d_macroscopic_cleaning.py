#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check electric cleaning in vacuum patches without removing material charge."""

import argparse

import numpy as np

from pywarpx import geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", choices=["vacuum", "conductor", "refined", "reject"], required=True
)
args = parser.parse_args()
geometry.dims = 3
refined = args.mode == "refined"
h = 1 / (64 if refined else 32)
c = 299792458.0
dt = 0.2 * h / c
steps = 1 if refined else 20
params = [
    "geometry.dims=3",
    "geometry.prob_lo=0 0 0",
    "geometry.prob_hi=1 0.25 0.25",
    "amr.n_cell=32 8 8",
    f"amr.max_level={int(refined)}",
    "amr.max_grid_size=32",
    "amr.blocking_factor=8",
    "amr.n_error_buf=0",
    "boundary.field_lo=periodic periodic periodic",
    "boundary.field_hi=periodic periodic periodic",
    "algo.maxwell_solver=yee",
    "algo.em_solver_medium=macroscopic",
    "algo.macroscopic_sigma_method=backwardeuler",
    "macroscopic.epsilon=epsilon0",
    "macroscopic.mu=mu0",
    "warpx.do_dive_cleaning=1",
    "warpx.use_filter=0",
    f"warpx.const_dt={dt}",
    f"max_step={steps}",
    "warpx.verbose=0",
    "amrex.the_arena_init_size=0",
    "warpx.E_ext_grid_init_style=parse_e_ext_grid_function",
    "warpx.Ex_external_grid_function(x,y,z)=cos(2*pi*x)",
    "warpx.Ey_external_grid_function(x,y,z)=0",
    "warpx.Ez_external_grid_function(x,y,z)=0",
    "amrex.abort_on_unused_inputs=1",
]
sigma = f"0.1*epsilon0/{dt}"
if args.mode in ("conductor", "reject"):
    params += [f"macroscopic.sigma={sigma}"]
elif refined:
    params += [
        f"macroscopic.sigma_function(x,y,z)=if((x<0.125)+(x>0.875)>0,{sigma},0)",
        "amr.ref_ratio=2",
        "warpx.fine_tag_lo=0.25 0 0",
        "warpx.fine_tag_hi=0.75 0.25 0.25",
        "warpx.pml_ncell=4",
        "warpx.pml_delta=4",
    ]
else:
    params += ["macroscopic.sigma=0"]
if args.mode in ("conductor", "refined"):
    params += ["macroscopic.dive_cleaning_vacuum_patches_only=1"]
libwarpx.initialize(["macroscopic-cleaning", *params])
libwarpx.warpx.evolve(steps)
if args.mode == "reject":
    raise AssertionError(
        "Non-vacuum electric cleaning must require the explicit patch restriction"
    )

register = libwarpx.warpx.multifab_register()
patches = [(0, "fp", 1 / 32)] if not refined else [(1, "fp", 1 / 64), (1, "cp", 1 / 32)]
for level, patch, spacing in patches:
    lam = 2 * c * dt / spacing * np.sin(np.pi * spacing)
    theta = 2 * np.arcsin(lam / 2)
    for family, direction in [("Efield", "x"), ("F", None)]:
        name = family + "_" + patch
        field = (
            register.get(name, dir=direction, level=level)
            if direction
            else register.get(name, level=level)
        )
        for mfi in field:
            valid, fab = mfi.validbox(), mfi.fabbox()
            region = tuple(
                slice(
                    valid.small_end[d] - fab.small_end[d],
                    valid.big_end[d] - fab.small_end[d] + 1,
                )
                for d in range(3)
            )
            data = field.array(mfi).to_numpy(copy=True, order="F")[region + (0,)]
            offset = 0.5 if direction else 0
            x = (np.arange(valid.small_end[0], valid.big_end[0] + 1) + offset) * spacing
            if args.mode == "conductor":
                expected = (
                    (1 / 1.1) ** steps * np.cos(2 * np.pi * x)
                    if direction
                    else np.zeros_like(x)
                )
            elif direction:
                expected = np.cos(steps * theta) * np.cos(2 * np.pi * x)
            else:
                data *= c
                expected = (
                    -np.sqrt(1 - lam**2 / 4)
                    * np.sin(steps * theta)
                    * np.sin(2 * np.pi * x)
                )
            if refined:
                # Keep the analytic stencil away from the initial PML interface.
                interior = (x > 0.375) & (x < 0.625)
                data, expected = data[interior], expected[interior]
            np.testing.assert_allclose(
                data,
                np.broadcast_to(expected[:, None, None], data.shape),
                rtol=64 * np.finfo(data.dtype).eps,
                atol=64 * np.finfo(data.dtype).eps,
            )
if refined:
    parent_f = register.get("F_fp", level=0)
    for mfi in parent_f:
        valid, fab = mfi.validbox(), mfi.fabbox()
        region = tuple(
            slice(
                valid.small_end[d] - fab.small_end[d],
                valid.big_end[d] - fab.small_end[d] + 1,
            )
            for d in range(3)
        )
        values = parent_f.array(mfi).to_numpy(copy=True, order="F")[region]
        assert np.count_nonzero(values) == 0, (
            "Material-bearing parent must not use particle-only Gauss cleaning"
        )
libwarpx.finalize()
