#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check parent-only initial fields and their SMR substitution in vacuum."""

import argparse

import numpy as np

from pywarpx import geometry, libwarpx

parser = argparse.ArgumentParser()
parser.add_argument("--style", choices=("constant", "parser"), required=True)
parser.add_argument("--levels", type=int, choices=(0, 1), required=True)
args = parser.parse_args()

geometry.dims = 3
inputs = [
    "initial-field-level",
    "geometry.dims=3",
    "geometry.prob_lo=-0.5 -0.5 -0.5",
    "geometry.prob_hi=0.5 0.5 0.5",
    "amr.n_cell=16 16 16",
    f"amr.max_level={args.levels}",
    "amr.max_grid_size=16",
    "amr.blocking_factor=4",
    "amr.n_error_buf=0",
    "boundary.field_lo=periodic periodic periodic",
    "boundary.field_hi=periodic periodic periodic",
    "warpx.const_dt=5.e-11",
    "max_step=64",
    "warpx.do_subcycling=0",
    "algo.maxwell_solver=yee",
    "warpx.use_filter=0",
    "warpx.maxlevel_extEMfield_init=0",
    "amrex.the_arena_init_size=0",
    "amrex.abort_on_unused_inputs=1",
    "warpx.verbose=0",
]
if args.levels:
    inputs += [
        "warpx.fine_tag_lo=-0.25 -0.25 -0.25",
        "warpx.fine_tag_hi=0.25 0.25 0.25",
        "warpx.pml_ncell=4",
    ]

initial_fields = {"E": (0.1, 0.2, 0.3), "B": (0.05, -0.1, 0.15)}
for field, values in initial_fields.items():
    if args.style == "constant":
        inputs += [
            f"warpx.{field}_ext_grid_init_style=constant",
            f"warpx.{field}_external_grid=" + " ".join(map(str, values)),
        ]
    else:
        inputs.append(
            f"warpx.{field}_ext_grid_init_style=parse_{field}_ext_grid_function"
        )
        inputs += [
            f"warpx.{field}{direction}_external_grid_function(x,y,z)={value}"
            for direction, value in zip("xyz", values)
        ]


def check_fields(*, check_auxiliary):
    """A uniform field is stationary; refined patch corrections stay zero."""
    register = libwarpx.warpx.multifab_register()
    for level in range(args.levels + 1):
        patches = ["fp"] if level == 0 else ["fp", "cp"]
        if check_auxiliary:
            patches.append("aux")
        for patch in patches:
            for field, values in initial_fields.items():
                for direction, value in zip("xyz", values):
                    expected = value if level == 0 or patch == "aux" else 0.0
                    mf = register.get(
                        f"{field}field_{patch}", dir=direction, level=level
                    )
                    np.testing.assert_allclose(
                        mf[...],
                        expected,
                        rtol=0.0,
                        atol=1.0e-12,
                        err_msg=f"level {level} {patch} {field}{direction}",
                    )


libwarpx.initialize(inputs)
check_fields(check_auxiliary=False)
# Cover propagation across the refined patch and its absorbing layer. The
# auxiliary field must contain the parent field exactly once at every step.
for _ in range(64):
    libwarpx.warpx.evolve(1)
    check_fields(check_auxiliary=True)
libwarpx.finalize()
