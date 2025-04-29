#!/usr/bin/env python3

# Copyright 2019-2021 Yinjian Zhao
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This script tests the collision module in RZ.
# Electrons are set with the same vr,
# and thus particles are traveling locally in the same direction.
# Physically, the collision rate should therefore be very low.
# Thus the initial electron momentum will be compared with the final momentum.

# Possible errors:
# tolerance: 1.0e-30
# Possible running time: ~ 1.0 s

import sys
from glob import glob

import numpy as np
import yt

tolerance = 1.0e-15

last_fn = sys.argv[1]
if last_fn[-1] == "/":
    last_fn = last_fn[:-1]
fn_list = glob(last_fn[:-5] + "?????")

for fn in fn_list:
    # get time index j
    j = int(fn[-5:])
    if j == 0:
        # load file
        ds = yt.load(fn)
        ad = ds.all_data()
        px1 = ad["particle_momentum_x"].to_ndarray()
        py1 = ad["particle_momentum_y"].to_ndarray()
    if j == 150:
        # load file
        ds = yt.load(fn)
        ad = ds.all_data()
        px2 = ad["particle_momentum_x"].to_ndarray()
        py2 = ad["particle_momentum_y"].to_ndarray()

error = np.max(abs(px1 - px2) + abs(py1 - py2))

print("error = ", error)
print("tolerance = ", tolerance)
assert error < tolerance
