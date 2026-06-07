#!/usr/bin/env python3

# Copyright 2026 Axel Huebl
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# Compare a plotfile of the runtime-dimensionality prototype (warpx.unified,
# degenerate-3D layout) against a plotfile of a native per-dimension WarpX
# binary (warpx.1d/2d/3d), field by field.
#
# Both codes cell-center the staggered fields with the exact same averaging
# (ablastr::coarsen::sample::Interp), so serial (1 rank, 1 thread) runs must
# agree bitwise; parallel runs agree up to atomic-reduction rounding.
#
# Usage: compare_native.py <unified_plotfile> <native_plotfile> [tolerance]
#        tolerance: maximum relative L-inf error (default 0.0, i.e., bitwise)
import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(50)

fn_unified = sys.argv[1]
fn_native = sys.argv[2]
tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

ds_u = yt.load(fn_unified)
ds_n = yt.load(fn_native)

data_u = ds_u.covering_grid(
    level=0, left_edge=ds_u.domain_left_edge, dims=ds_u.domain_dimensions
)
data_n = ds_n.covering_grid(
    level=0, left_edge=ds_n.domain_left_edge, dims=ds_n.domain_dimensions
)

assert abs(ds_u.current_time.to_value() - ds_n.current_time.to_value()) == 0.0, (
    f"simulation times differ: {ds_u.current_time} vs {ds_n.current_time}"
)

fields = ["Ex", "Ey", "Ez", "Bx", "By", "Bz", "jx", "jy", "jz"]
failed = []
for field in fields:
    f_u = np.squeeze(data_u[("boxlib", field)].to_ndarray())
    try:
        f_n = np.squeeze(data_n[("boxlib", field)].to_ndarray())
    except Exception:
        print(f"{field}: not in native output, skipped")
        continue
    assert f_u.shape == f_n.shape, f"{field}: shape mismatch {f_u.shape} vs {f_n.shape}"
    norm = np.abs(f_n).max()
    err = np.abs(f_u - f_n).max()
    rel = err / norm if norm > 0.0 else err
    exact = " (bitwise)" if err == 0.0 else ""
    print(f"{field}: max |diff| = {err:.3e}, rel = {rel:.3e}{exact}")
    if rel > tolerance:
        failed.append(field)

if failed:
    print(f"FAIL: {failed} exceed tolerance {tolerance}")
    sys.exit(1)
print("PASS")
