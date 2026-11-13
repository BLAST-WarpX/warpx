#!/usr/bin/env python3
"""Integrate radial current over a cylindrical shell cut by an axial refinement boundary."""

import sys

import numpy as np
import yt

ds = yt.load(sys.argv[1])
assert ds.index.max_level == 1
r0, r1 = 0.4375, 0.5625
parts = np.zeros(2)
parent_shells = []
for grid in ds.index.grids:
    dr, dz = np.asarray(grid.dds)[:2]
    radius = float(grid.LeftEdge[0]) + (np.arange(grid.ActiveDimensions[0]) + 0.5) * dr
    selected = ((radius >= r0) & (radius < r1))[:, None]
    leaf = np.asarray(grid.child_mask, dtype=bool)[:, :, 0]
    jr = np.asarray(grid[("boxlib", "jr")])[:, :, 0]
    if grid.Level == 0:
        # The synchronized parent includes the restricted source under the patch.
        # Each shell in this single radial box crosses the two axial interfaces.
        parent_shells.extend((2 * np.pi * radius * dz * jr.sum(axis=1))[selected[:, 0]])
    else:
        z = float(grid.LeftEdge[1]) + (np.arange(grid.ActiveDimensions[1]) + 0.5) * dz
        interior = selected & (abs(z) < 0.1)[None, :]
        np.testing.assert_allclose(
            (2 * np.pi * radius[:, None] * jr)[interior], 1500 / 0.5, rtol=2e-6
        )
    parts[grid.Level] += (jr * leaf * selected * radius[:, None]).sum() * (
        2 * np.pi * dr * dz
    )

# Divide the annular volume integral by its radial width to obtain amperes.
parts /= r1 - r0
print(f"Radial current contributions from the two leaves (A): {parts}")
assert len(parent_shells) > 0
np.testing.assert_allclose(parent_shells, 1500.0, rtol=2e-6)
assert np.all(parts > 0)
# The cell-centered leaf plot samples the edge stencil once on each side of
# the interface. At this resolution its annular integral is within one percent.
np.testing.assert_allclose(parts.sum(), 1500.0, rtol=0.01)
