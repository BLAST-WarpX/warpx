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

# Possible errors: 6.235230443866285e-9
# tolerance: 1.0e-8
# Possible running time: 0.327827743 s

import os
import sys

import numpy as np
import yt

sys.path.insert(1, "../../../../warpx/Regression/Checksum/")
from checksumAPI import evaluate_checksum

tolerance = 1.0e-8
r0 = 0.12402005
z0 = 4.3632492

filename = sys.argv[1]
ds = yt.load(filename)
ad = ds.all_data()
r = ad["proton", "particle_position_x"].to_ndarray()
z = ad["proton", "particle_position_y"].to_ndarray()

error = np.min(np.sqrt((r - r0) ** 2 + (z - z0) ** 2))

print("error = ", error)
print("tolerance = ", tolerance)
assert error < tolerance

# compare checksums
evaluate_checksum(
    test_name=os.path.split(os.getcwd())[1],
    output_file=sys.argv[1],
)
