#!/usr/bin/env python3
"""Compare the charge and axial current of a uniform drifting plasma across CPU tiles."""

import sys

import numpy as np
import yt
from scipy.constants import c, elementary_charge

ds = yt.load(sys.argv[1])
rho = -1.0e9 * elementary_charge
jz = rho * c * 0.125 / np.sqrt(1 + 0.125**2)
for grid in ds.index.grids:
    dr = float(grid.dds[0])
    r = float(grid.LeftEdge[0]) + (np.arange(grid.ActiveDimensions[0]) + 0.5) * dr
    # The deposition in the interior samples a radially weighted, constant
    # density. All axial rows are included, including shared nodal tile faces.
    interior = (r > 0.1) & (r < 0.9)
    for name, expected in [("rho_electrons", rho), ("jz", jz)]:
        actual = np.asarray(grid[("boxlib", name)])[interior]
        np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=0)
