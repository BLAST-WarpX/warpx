#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check the damping on coplanar and separated faces sharing a PML box."""

import argparse

import numpy as np

from pywarpx import callbacks, geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument("--max-grid-size", type=int, required=True)
parser.add_argument("--frame", action="store_true")
args = parser.parse_args()
geometry.dims = 3
axis = 0 if args.frame else 2
split = 1 if args.frame else 0  # Eyx or Eyz


def seed_pml():
    # Match the regular Ey = 1 with one unit split component, including guards. With
    # initially zero B and current, the first E update is pure PML damping.
    register = libwarpx.warpx.multifab_register()
    for patch in ("fp", "cp"):
        field = register.get(f"pml_E_{patch}", dir="y", level=1)
        field.set_val(1.0, split, 1, field.n_grow_vect)


refinement_region = "if(x*x+y*y < 0.0265*0.0265, if(abs(z)<0.0416666666666667,1,0),0)"
if args.frame:
    refinement_region = (
        "if((abs(x)<0.042)*(abs(y)<0.042)*((abs(x)>0.0225)+(abs(y)>0.0225)),1,0)"
    )
callbacks.installcallback("allocdata", seed_pml)
libwarpx.initialize(
    [
        "pml-coincident-faces",
        "geometry.dims=3",
        "geometry.prob_lo=-0.05 -0.05 -0.05",
        "geometry.prob_hi=0.05 0.05 0.05",
        "amr.n_cell=96 96 16" if args.frame else "amr.n_cell=48 48 48",
        "amr.max_level=1",
        "amr.ref_ratio_vect=2 2 1" if args.frame else "amr.ref_ratio=2",
        f"amr.max_grid_size={args.max_grid_size}",
        "amr.blocking_factor=8" if args.frame else "amr.blocking_factor=4",
        "amr.n_error_buf=0",
        "amr.grid_eff=1",
        f"warpx.ref_patch_function(x,y,z)={refinement_region}",
        "boundary.field_lo=periodic periodic periodic",
        "boundary.field_hi=periodic periodic periodic",
        "warpx.const_dt=5e-13",
        "max_step=1",
        "warpx.do_subcycling=0",
        "warpx.pml_ncell=4" if args.frame else "warpx.pml_ncell=2",
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

parent_cells = 96 if args.frame else 48
width = 4 if args.frame else 2
for patch, ncell, delta, spacing in (
    ("fp", width, 4, 0.1 / (2 * parent_cells)),
    ("cp", width // 2, 2, 0.1 / parent_cells),
):
    regular = register.get(f"Efield_{patch}", dir="y", level=1)
    boxes = regular.box_array()
    nodal = np.array([boxes.ix_type().node_centered(d) for d in range(3)])
    grids = [(np.array(box.small_end), np.array(box.big_end) - nodal) for box in boxes]
    field = register.get(f"pml_E_{patch}", dir="y", level=1)
    transverse = np.arange(3) != axis
    checked = 0
    for mfi in field:
        valid = mfi.tilebox()
        low = np.array(valid.small_end)
        high = np.array(valid.big_end)
        cell_high = high - nodal
        faces = []
        for grid_low, grid_high in grids:
            if np.any(grid_low[transverse] > cell_high[transverse]) or np.any(
                grid_high[transverse] < low[transverse]
            ):
                continue
            if grid_low[axis] - ncell <= cell_high[axis] < grid_low[axis]:
                faces.append((int(grid_low[axis]), -1))
            elif grid_high[axis] < low[axis] <= grid_high[axis] + ncell:
                faces.append((int(grid_high[axis] + 1), 1))
        if len(faces) < 2:
            continue
        profiles = set(faces)
        if args.frame:
            if {side for _, side in profiles} != {-1, 1}:
                continue
        else:
            assert len(profiles) == 1
        origin = np.array(mfi.fabbox().small_end)
        region = tuple(
            slice(low[d] - origin[d], high[d] - origin[d] + 1) for d in range(3)
        )
        split_values = field.array(mfi).to_numpy(copy=True, order="F")[region]
        # At a nodal interface the regular field can enter the first split
        # component. Its physical tangential value is the sum of the splits.
        values = split_values.sum(axis=-1) if args.frame else split_values[..., split]
        coordinate = np.arange(low[axis], high[axis] + 1)
        sigma = np.zeros(len(coordinate))
        for plane, side in profiles:
            distance = side * (coordinate - plane)
            active = (distance >= 0) & (distance <= ncell)
            sigma[active] = (
                4 * 299792458.0 / (spacing * delta**2) * distance[active] ** 2
            )
        shape = [1, 1, 1]
        shape[axis] = -1
        expected = np.broadcast_to(np.exp(-sigma * 5e-13).reshape(shape), values.shape)
        np.testing.assert_allclose(
            values, expected, rtol=32 * np.finfo(values.dtype).eps, atol=0.0
        )
        checked += 1
    assert checked > 0, f"No matching {patch} damping profiles were exercised"
    print(f"Checked {checked} {patch} PML profiles")

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
