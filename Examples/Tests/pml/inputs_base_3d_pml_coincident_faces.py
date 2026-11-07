#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check identical damping profiles shared by coplanar refinement-box faces."""

import argparse

import numpy as np

from pywarpx import callbacks, geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument("--max-grid-size", type=int, required=True)
args = parser.parse_args()
geometry.dims = 3


def seed_pml():
    # Match the regular Ey = 1 with Eyz = 1, including guard cells. With
    # initially zero B and current, the first E update is pure PML damping.
    register = libwarpx.warpx.multifab_register()
    for patch in ("fp", "cp"):
        field = register.get(f"pml_E_{patch}", dir="y", level=1)
        field.set_val(1.0, 0, 1, field.n_grow_vect)


callbacks.installcallback("allocdata", seed_pml)
libwarpx.initialize(
    [
        "pml-coincident-faces",
        "geometry.dims=3",
        "geometry.prob_lo=-0.05 -0.05 -0.05",
        "geometry.prob_hi=0.05 0.05 0.05",
        "amr.n_cell=48 48 48",
        "amr.max_level=1",
        "amr.ref_ratio=2",
        f"amr.max_grid_size={args.max_grid_size}",
        "amr.blocking_factor=4",
        "amr.n_error_buf=0",
        "amr.grid_eff=1",
        "warpx.ref_patch_function(x,y,z)="
        "if(x*x+y*y < 0.0265*0.0265, if(abs(z)<0.0416666666666667,1,0),0)",
        "boundary.field_lo=periodic periodic periodic",
        "boundary.field_hi=periodic periodic periodic",
        "warpx.const_dt=5e-13",
        "max_step=1",
        "warpx.do_subcycling=0",
        "warpx.pml_ncell=2",
        "warpx.pml_delta=4",
        "algo.maxwell_solver=yee",
        "warpx.use_filter=0",
        "warpx.E_ext_grid_init_style=constant",
        "warpx.E_external_grid=0 1 0",
        "warpx.verbose=0",
        "amrex.the_arena_init_size=0",
        "amrex.abort_on_unused_inputs=1",
    ]
)
libwarpx.warpx.evolve(1)
register = libwarpx.warpx.multifab_register()

for patch, ncell, delta, spacing in (("fp", 2, 4, 0.1 / 96), ("cp", 1, 2, 0.1 / 48)):
    regular = register.get(f"Efield_{patch}", dir="y", level=1)
    boxes = regular.box_array()
    nodal = np.array([boxes.ix_type().node_centered(d) for d in range(3)])
    grids = [(np.array(box.small_end), np.array(box.big_end) - nodal) for box in boxes]
    field = register.get(f"pml_E_{patch}", dir="y", level=1)
    checked = 0
    for mfi in field:
        valid = mfi.tilebox()
        low = np.array(valid.small_end)
        high = np.array(valid.big_end)
        cell_high = high - nodal
        planes = []
        for grid_low, grid_high in grids:
            if np.any(grid_low[:2] > cell_high[:2]) or np.any(grid_high[:2] < low[:2]):
                continue
            if grid_low[2] - ncell <= cell_high[2] < grid_low[2]:
                planes.append(int(grid_low[2]))
            elif grid_high[2] < low[2] <= grid_high[2] + ncell:
                planes.append(int(grid_high[2] + 1))
        if len(planes) < 2:
            continue
        assert len(set(planes)) == 1, "The fixture must request identical profiles"
        origin = np.array(mfi.fabbox().small_end)
        region = tuple(
            slice(low[d] - origin[d], high[d] - origin[d] + 1) for d in range(3)
        )
        values = field.array(mfi).to_numpy(copy=True, order="F")[region + (0,)]
        z = np.arange(low[2], high[2] + 1)
        sigma = 4 * 299792458.0 / (spacing * delta**2) * (z - planes[0]) ** 2
        expected = np.broadcast_to(np.exp(-sigma * 5e-13), values.shape)
        np.testing.assert_allclose(
            values, expected, rtol=32 * np.finfo(values.dtype).eps, atol=0.0
        )
        checked += 1
    assert checked > 0, f"No coincident {patch} damping profiles were exercised"
    print(f"Checked {checked} coincident {patch} PML profiles")

# Exercise every damping direction, including zero split components. Undefined
# factors would turn those components into NaNs even for this uniform fixture.
for patch in ("fp", "cp"):
    for family in ("E", "B"):
        for direction in "xyz":
            field = register.get(f"pml_{family}_{patch}", dir=direction, level=1)
            for mfi in field:
                valid = mfi.tilebox()
                origin = mfi.fabbox().small_end
                region = tuple(
                    slice(
                        valid.small_end[d] - origin[d], valid.big_end[d] - origin[d] + 1
                    )
                    for d in range(3)
                )
                values = field.array(mfi).to_numpy(copy=True, order="F")[region]
                assert np.isfinite(values).all(), (patch, family, direction)

libwarpx.finalize()
