#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Verify the discrete dispersion of a longitudinal magnetic cleaning wave."""

import numpy as np

from pywarpx import callbacks, geometry, libwarpx

geometry.dims = 3
c = 299792458.0
h = 1 / 32
dt = 0.2 * h / c
steps = 20


def seed():
    field = libwarpx.warpx.multifab_register().get("G_fp", level=0)
    for mfi in field:
        data = field.array(mfi).to_xp(copy=False, order="F")
        for i in range(data.shape[0]):
            x = (mfi.fabbox().small_end[0] + i + 0.5) * h
            data[i, :, :, 0] = np.cos(2 * np.pi * x)


callbacks.installcallback("allocdata", seed)
libwarpx.initialize(
    [
        "magnetic-cleaning-wave",
        "geometry.dims=3",
        "geometry.prob_lo=0 0 0",
        "geometry.prob_hi=1 0.25 0.25",
        "amr.n_cell=32 8 8",
        "amr.max_level=0",
        "amr.max_grid_size=16",
        "amr.blocking_factor=8",
        "boundary.field_lo=periodic periodic periodic",
        "boundary.field_hi=periodic periodic periodic",
        "algo.maxwell_solver=yee",
        "warpx.do_divb_cleaning=1",
        "warpx.use_filter=0",
        f"warpx.const_dt={dt}",
        f"max_step={steps}",
        "warpx.verbose=0",
        "amrex.the_arena_init_size=0",
        "amrex.abort_on_unused_inputs=1",
    ]
)
libwarpx.warpx.evolve(steps)

# Yee dispersion: sin(omega*dt/2) = c*dt/h*sin(k*h/2).
# G lives at electric times; B is synchronized by two magnetic half pushes.
lam = 2 * c * dt / h * np.sin(np.pi * h)
theta = 2 * np.arcsin(lam / 2)
register = libwarpx.warpx.multifab_register()
for name, direction in [("G_fp", None), ("Bfield_fp", "x")]:
    field = (
        register.get(name, dir=direction, level=0)
        if direction
        else register.get(name, level=0)
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
        values = field.array(mfi).to_numpy(copy=True, order="F")[region + (0,)]
        offset = 0 if direction else 0.5
        x = (np.arange(valid.small_end[0], valid.big_end[0] + 1) + offset) * h
        if direction:
            values = c * values
            expected = (
                -np.sqrt(1 - lam**2 / 4) * np.sin(steps * theta) * np.sin(2 * np.pi * x)
            )
        else:
            expected = np.cos(steps * theta) * np.cos(2 * np.pi * x)
        np.testing.assert_allclose(
            values,
            np.broadcast_to(expected[:, None, None], values.shape),
            rtol=64 * np.finfo(values.dtype).eps,
            atol=64 * np.finfo(values.dtype).eps,
        )
libwarpx.finalize()
