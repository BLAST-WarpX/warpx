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
#
# An optional second argument overrides the tolerance, for variants that use
# different solvers. The theta-implicit variant differs from the explicit
# reference at second order in dt; possible errors: 5.3e-4.

import sys

import numpy as np
import yt

tolerance = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0e-8
r0 = 0.12402005
z0 = 4.3632492

filename = sys.argv[1]
ds = yt.load(filename)
# The proton will leave the simulation if no external field is applied
n_protons = ds.particle_type_counts.get("proton", 0)
assert n_protons == 1, f"expected 1 proton, found {n_protons}"

ad = ds.all_data()
r = ad["proton", "particle_position_x"].to_ndarray()
z = ad["proton", "particle_position_y"].to_ndarray()

error = np.min(np.sqrt((r - r0) ** 2 + (z - z0) ** 2))

print("error = ", error)
print("tolerance = ", tolerance)
assert error < tolerance
