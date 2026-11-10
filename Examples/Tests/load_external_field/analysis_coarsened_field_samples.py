#!/usr/bin/env python3
"""Check the sample positions of all Yee field components in a coarsened plotfile."""

import sys

import numpy as np
import yt

ds = yt.load(sys.argv[1])
nx, nz, _ = ds.domain_dimensions
x = (np.arange(nx)[:, None] + 0.5) / nx
z = (np.arange(nz)[None, :] + 0.5) / nz
grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
fields = {
    "Ex": 1 + x + 2 * z,
    "Ey": 2 - 2 * x + z,
    "Ez": 3 + 3 * x - z,
    "Bx": 4 - x + 2 * z,
    "By": 5 + 2 * x + z,
    "Bz": 6 - x - 3 * z,
}
# The analytic fields are affine in the interior and wrap at the periodic boundaries.
for name, expected in fields.items():
    actual = np.asarray(grid["boxlib", name])[:, :, 0]
    np.testing.assert_allclose(
        actual[1:-1, 1:-1], expected[1:-1, 1:-1], rtol=2e-6, atol=2e-6, err_msg=name
    )
