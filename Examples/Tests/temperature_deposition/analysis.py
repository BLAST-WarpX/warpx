#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Check native deposited temperatures against the exact two-sample identity.

For any two positive effective weights (macro weight times shape), the
unbiased weighted variance is (v1-v2)^2/2. Thus these checks do not reproduce
WarpX's sum-of-squared-weights implementation. Species exercise equal and
unequal macro weights, equal macro/unequal shape weights, 50 decades of common
weight rescaling, RZ axis support, and empty/single effective-sample behavior.
The selected cells have both particles in every staggered interpolation point.
"""

import sys

import numpy as np
import yt
from scipy.constants import k, m_p

# Particle speeds are (1,2,3) m/s, opposite in each pair. Relativistic
# corrections are below roundoff. Native fields are temperature in kelvin.
expected = 2.0 * np.array([1.0, 4.0, 9.0]) * m_p / k
cases = ["equal", "macro", "shape", "small", "large", "axis", "nearaxis"]
ds = yt.load(sys.argv[1])
grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
is_rz = ("boxlib", "Tr_equal") in ds.field_list

for name in cases + ["single", "zero", "dominant"]:
    for direction, component in enumerate("rtz" if is_rz else "xyz"):
        temperature = grid["boxlib", f"T{component}_{name}"].to_ndarray()
        assert np.isfinite(temperature).all(), f"nonfinite T{component}_{name}"
        if name in ("single", "zero", "dominant"):
            assert np.max(np.abs(temperature)) == 0.0, name
            continue
        r = 0 if name in ("axis", "nearaxis") else 8
        point = (r, 8, 0) if is_rz else (r, 8, 8)
        measured = float(temperature[point])
        ratio = measured / expected[direction]
        print(f"{name} T{component}: measured/exact = {ratio:.12g}")
        # A deterministic identity, allowing single-precision deposition.
        # Old n/(n-1) gives 0.36 for the 1:9 pair and also fails shape-only.
        assert np.isclose(ratio, 1.0, rtol=5.0e-5, atol=0.0), (name, component, ratio)

print("Weighted particle temperature regression: PASS")
