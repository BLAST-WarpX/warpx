#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Rigid material translation must preserve its uniform caloric temperature."""

from pathlib import Path

import numpy as np
import yt

yt.funcs.mylog.setLevel(50)
plots = sorted(
    path for path in Path("diags").glob("plt[0-9]*") if path.name[3:].isdigit()
)
assert len(plots) == 2
first, last = [yt.load(str(path)) for path in plots]
assert float(last.current_time) > 0
grids = [
    ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    for ds in (first, last)
]
temperature = [np.asarray(grid["boxlib", "Te"]) for grid in grids]
relative_error = np.max(abs(temperature[1] - temperature[0])) / np.max(temperature[0])
assert relative_error < 1.0e-9, relative_error
rho = [np.asarray(grid["boxlib", "rho"]) for grid in grids]
# Ensure an actual translating, nonuniform deposit exercised the update.
relative_motion = np.max(abs(rho[1] - rho[0])) / np.max(abs(rho[0]))
assert relative_motion > 0.05, relative_motion
print(f"Uniform-temperature relative error: {relative_error:.6g}")
print(f"Nonuniform charge-profile change: {relative_motion:.6g}")
