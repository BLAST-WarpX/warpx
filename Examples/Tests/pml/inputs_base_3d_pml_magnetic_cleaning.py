#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check independent regular-grid and PML magnetic cleaning with Cartesian Yee and CKC solvers."""

import argparse

import numpy as np

from pywarpx import callbacks, geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument("--regular", type=int, required=True)
parser.add_argument("--pml", type=int, required=True)
parser.add_argument("--refined", type=int, required=True)
parser.add_argument("--seed-g", action="store_true")
parser.add_argument("--solver", choices=["yee", "ckc"], default="yee")
args = parser.parse_args()
geometry.dims = 3
dt = 1e-12


def seed_pml():
    register = libwarpx.warpx.multifab_register()
    patches = [("fp", 0, 1 / 16)]
    if args.refined:
        patches += [("fp", 1, 1 / 32), ("cp", 1, 1 / 16)]
    for patch, level, h in patches:
        if args.seed_g:
            assert args.pml
            field = register.get(f"pml_G_{patch}", level=level)
            for mfi in field:
                data = field.array(mfi).to_xp(copy=False, order="F")
                for i in range(data.shape[0]):
                    data[i, :, :, 0] = (mfi.fabbox().small_end[0] + i + 0.5) * h
            continue
        for direction in ("x", "z"):
            field = register.get(f"pml_B_{patch}", dir=direction, level=level)
            offset = 0 if field.box_array().ix_type().node_centered(0) else 0.5
            # Bx = x tests div(B), while Bzz = x tests the curl contribution
            # from the third magnetic PML component when cleaning is enabled.
            component = 2 if direction == "z" and args.pml else 0
            for mfi in field:
                data = field.array(mfi).to_xp(copy=False, order="F")
                for i in range(data.shape[0]):
                    data[i, :, :, component] = (
                        mfi.fabbox().small_end[0] + i + offset
                    ) * h


callbacks.installcallback("allocdata", seed_pml)
libwarpx.initialize(
    [
        "pml-magnetic-cleaning",
        "geometry.dims=3",
        "geometry.prob_lo=0 0 0",
        "geometry.prob_hi=1 1 1",
        "amr.n_cell=16 16 16",
        f"amr.max_level={args.refined}",
        "amr.max_grid_size=32",
        "amr.blocking_factor=8",
        "amr.n_error_buf=0",
        "boundary.field_lo=pml periodic periodic",
        "boundary.field_hi=pml periodic periodic",
        f"warpx.const_dt={dt}",
        "max_step=1",
        "warpx.do_subcycling=0",
        "warpx.pml_ncell=8",
        "warpx.pml_delta=8",
        f"warpx.do_divb_cleaning={args.regular}",
        f"warpx.do_pml_divb_cleaning={args.pml}",
        f"algo.maxwell_solver={args.solver}",
        "warpx.use_filter=0",
        "warpx.do_initial_div_cleaning=0",
        "warpx.B_ext_grid_init_style=parse_b_ext_grid_function",
        f"warpx.Bx_external_grid_function(x,y,z)={0 if args.seed_g else 'x'}",
        "warpx.By_external_grid_function(x,y,z)=0",
        f"warpx.Bz_external_grid_function(x,y,z)={0 if args.seed_g else 'x'}",
        "warpx.verbose=0",
        "amrex.the_arena_init_size=0",
        "amrex.abort_on_unused_inputs=1",
    ]
    + (
        ["amr.ref_ratio=2", "warpx.fine_tag_lo=0.25 0 0", "warpx.fine_tag_hi=0.75 1 1"]
        if args.refined
        else []
    )
)
libwarpx.warpx.evolve(1)
register = libwarpx.warpx.multifab_register()
patches = [("fp", 0, 1 / 16, 8, (0, 16))]
if args.refined:
    patches += [("fp", 1, 1 / 32, 8, (8, 24)), ("cp", 1, 1 / 16, 4, (4, 12))]
for patch, level, h, delta, planes in patches:
    fields = [("pml_B", "x")] if args.seed_g else [("pml_E", "y")]
    if args.pml and not args.seed_g:
        fields += [("pml_G", None)]
    for family, direction in fields:
        name = f"{family}_{patch}"
        field = (
            register.get(name, dir=direction, level=level)
            if direction
            else register.get(name, level=level)
        )
        checked = 0
        for mfi in field:
            valid, fab = mfi.validbox(), mfi.fabbox()
            data = field.array(mfi).to_numpy(copy=True, order="F")
            assert np.isfinite(data).all(), (name, level)
            # Exclude boundary stencils, where G meets the regular grid.
            margin = 2 if args.seed_g else 1
            low = np.array(valid.small_end) + margin
            high = np.array(valid.big_end) - margin
            if np.any(high < low):
                continue
            region = tuple(
                slice(low[d] - fab.small_end[d], high[d] - fab.small_end[d] + 1)
                for d in range(3)
            )
            offset = 0.5 if family == "pml_G" else 0
            x = np.arange(low[0], high[0] + 1) + offset
            plane = planes[0] if valid.big_end[0] <= planes[0] else planes[1]
            sigma = 4 * 299792458.0 / (h * delta**2) * (x - plane) ** 2
            sign = 1 if family == "pml_G" else -1
            amplitude = 1 if args.seed_g else sign * 299792458.0**2
            expected = amplitude * dt * np.exp(-sigma * dt)
            component = 2 if args.seed_g else (0 if family == "pml_G" else 1)
            values = data[region + (component,)]
            np.testing.assert_allclose(
                values,
                np.broadcast_to(expected[:, None, None], values.shape),
                rtol=64 * np.finfo(values.dtype).eps,
                atol=0,
            )
            checked += 1
        assert checked > 0, (name, level)

libwarpx.finalize()
