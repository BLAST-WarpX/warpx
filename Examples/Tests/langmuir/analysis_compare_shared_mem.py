#!/usr/bin/env python3

# Compare the fields of a simulation that uses the shared-memory deposition with
# those of the same simulation that does not: they must agree up to round-off errors.

import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(50)

filename = sys.argv[1]
filename_benchmark = sys.argv[2]
tolerance = 1.0e-9

ds = yt.load(filename)
ds_benchmark = yt.load(filename_benchmark)
assert ds.index.max_level == ds_benchmark.index.max_level

for lev in range(ds.index.max_level + 1):
    dims = ds.domain_dimensions * ds.refine_by**lev
    data = ds.covering_grid(lev, ds.domain_left_edge, dims)
    data_benchmark = ds_benchmark.covering_grid(lev, ds.domain_left_edge, dims)
    for field in ds.field_list:
        if field[0] != "boxlib":
            continue
        values = data[field].v
        values_benchmark = data_benchmark[field].v
        scale = np.max(np.abs(values_benchmark))
        if scale == 0.0:
            assert np.all(values == 0.0), f"level {lev}, {field[1]}"
            continue
        error = np.max(np.abs(values - values_benchmark)) / scale
        print(f"level {lev}, {field[1]}: relative error = {error:.3e}")
        assert error < tolerance, f"level {lev}, {field[1]}"
