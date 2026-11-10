#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""A fully refined vacuum domain must cancel the duplicate coarse Maxwell response."""

import argparse

import numpy as np

from pywarpx import geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument("--in-domain", action="store_true")
parser.add_argument("--directional", action="store_true")
parser.add_argument(
    "--reference", help="Scalar PML pulse used for the component-wise comparison"
)
parser.add_argument("--invalid", choices=("length", "value"))
args = parser.parse_args()
geometry.dims = 3
steps = 72
dt = 0.2 / (32 * 299792458.0)
libwarpx.initialize(
    [
        "coincident-physical-pml",
        "geometry.dims=3",
        "geometry.prob_lo=0 0 0",
        "geometry.prob_hi=0.25 0.25 1",
        "amr.n_cell=8 8 32",
        "amr.max_level=1",
        "amr.ref_ratio=2",
        "amr.max_grid_size=16",
        "amr.blocking_factor=8",
        "warpx.fine_tag_lo=0 0 0",
        "warpx.fine_tag_hi=0.25 0.25 1",
        "boundary.field_lo=periodic periodic pml",
        "boundary.field_hi=periodic periodic pml",
        f"warpx.pml_ncell={6 if args.directional else 4}",
        f"warpx.pml_delta={2 if args.directional else 4}",
        *(
            ["warpx.pml_ncell_vect=2 3 4", "warpx.pml_delta_vect=3 2 4"]
            if args.directional
            else []
        ),
        *(["warpx.pml_ncell_vect=2 4"] if args.invalid == "length" else []),
        *(["warpx.pml_delta_vect=3 0 4"] if args.invalid == "value" else []),
        f"warpx.do_pml_in_domain={int(args.in_domain)}",
        "algo.maxwell_solver=yee",
        "warpx.use_filter=0",
        f"warpx.const_dt={dt}",
        f"max_step={steps}",
        "warpx.verbose=0",
        "amrex.the_arena_init_size=0",
        "amrex.abort_on_unused_inputs=1",
    ]
)
fields = libwarpx.warpx.multifab_register()
# The z-oriented strips have the same physical cell count on the two duplicate
# coarse grids. Inspect their actual geometry independently of the Maxwell response.
for level, patch, nz in ((0, "fp", 32), (1, "cp", 32), (1, "fp", 64)):
    magnetic = fields.get(f"pml_B_{patch}", dir="x", level=level)
    axial_cells = set()
    for mfi in magnetic:
        box = mfi.validbox()
        axial_cells.update(range(box.small_end[2], box.big_end[2] + 1))
    expected = (
        set(range(4)) | set(range(nz - 4, nz))
        if args.in_domain
        else set(range(-4, 0)) | set(range(nz, nz + 4))
    )
    assert axial_cells == expected, (level, patch, axial_cells, expected)
# Launch a compact pulse toward both ends. The parent and coarse patch have
# identical spacing and initial data with different box decompositions.
for level, patch, nz in ((0, "fp", 32), (1, "cp", 32), (1, "fp", 64)):
    field = fields.get(f"Efield_{patch}", dir="y", level=level)
    for mfi in field:
        data = field.array(mfi).to_xp(copy=False, order="F")
        first = mfi.fabbox().small_end[2]
        for k in range(data.shape[2]):
            z = (first + k) / nz
            data[:, :, k, 0] = max(0.0, 1 - ((z - 0.72) / 0.12) ** 2) ** 3

libwarpx.warpx.evolve(steps)
response = {}
for family, direction in (("E", "y"), ("B", "x")):
    name = f"{family}field"
    parent = np.asarray(fields.get(name + "_fp", dir=direction, level=0)[...])
    patch = np.asarray(fields.get(name + "_cp", dir=direction, level=1)[...])
    scale = np.max(np.abs(parent))
    assert scale > (0.1 if family == "E" else 0.1 / 299792458.0)
    error = np.max(np.abs(parent - patch)) / scale
    print(f"{family} parent/coarse-patch relative difference: {error:.3e}")
    assert error <= 128 * np.finfo(parent.dtype).eps
    response[family] = parent.copy()
if args.reference:
    with np.load(args.reference) as scalar:
        for family, field in response.items():
            np.testing.assert_allclose(
                field,
                scalar[family],
                rtol=0,
                atol=128 * np.finfo(field.dtype).eps * np.max(abs(field)),
            )
else:
    np.savez("pml_response.npz", **response)
libwarpx.finalize()
