#!/usr/bin/env python3

# Copyright 2022 Yinjian Zhao
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This test tests the external field loading feature.
# A magnetic mirror field is loaded, and a single particle
# in the mirror will be reflected by the magnetic mirror effect.
# At the end of the simulation, the position of the particle
# is compared with known correct results.

# Possible errors: 3.915833656984999e-9
# tolerance: 1.0e-8
# Possible running time: 2.756646401 s

import os
import sys

import numpy as np
import yt

sys.path.insert(1, "../../../../warpx/Regression/Checksum/")
from checksumAPI import evaluate_checksum

tolerance = 1.0e-8
x0 = 0.12238072
y0 = 0.00965394
z0 = 4.31754477

filename = sys.argv[1]

ds = yt.load(filename)
ad = ds.all_data()
x = ad["proton", "particle_position_x"].to_ndarray()
y = ad["proton", "particle_position_y"].to_ndarray()
z = ad["proton", "particle_position_z"].to_ndarray()

error = np.min(np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2))

print("error = ", error)
print("tolerance = ", tolerance)
assert error < tolerance

# compare checksums
evaluate_checksum(
    test_name=os.path.split(os.getcwd())[1],
    output_file=sys.argv[1],
)
