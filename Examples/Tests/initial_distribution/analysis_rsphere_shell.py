#!/usr/bin/env python3

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

# Load particles
ts = OpenPMDTimeSeries("./diags/diag1")

x, y, z = ts.get_particle(
    ["x", "y", "z"],
    species="electrons",
    iteration=0,
)

# Cartesian coordinates -> spherical radius
r = np.sqrt(x**2 + y**2 + z**2)

# Check that the correct number of particles is loaded
assert len(r) == 304

# Check that the particles extend over the correct range
assert r.min() < 2.003
assert r.max() > 2.997
