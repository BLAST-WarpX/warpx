"""Recover the signed current of each proton from the radial Yee flux surfaces."""

import sys

import numpy as np
import yt
from scipy.constants import c, elementary_charge

ds = yt.load(sys.argv[1])
grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
jr = grid["boxlib", "jr"].to_value().ravel()
dr = ds.domain_width[0].to_value("m") / len(jr)
r = (np.arange(len(jr)) + 0.5) * dr
geometry = ds.parameters["geometry.dims"]
if geometry == "RCYLINDER":
    area = 2 * np.pi * r
    # The second particle also moves along the cylinder axis.
    ur = np.array([5.0e4, -4.0e4])
elif geometry == "RSPHERE":
    area = 4 * np.pi * r**2
    ur = np.array([5.0e4, -5.0e4])
else:
    raise ValueError(f"Unexpected radial geometry {geometry}")

current = [np.sum(slab) for slab in np.split(jr * area * dr, 2)]
expected = elementary_charge * 1.0e9 * ur / np.sqrt(1 + (5.0e4 / c) ** 2)
np.testing.assert_allclose(current, expected, rtol=1.0e-6, atol=0.0)
