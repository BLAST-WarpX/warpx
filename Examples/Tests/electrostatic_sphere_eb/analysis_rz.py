#!/usr/bin/env python

# Copyright 2019-2021 Yinjian Zhao
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This script tests the embedded boundary in RZ.
# A cylindrical surface (r=0.1) has a fixed potential 1 V.
# The outer surface has 0 V fixed.
# Thus the analytical solution has the form:
# phi(r) = A+B*log(r), Er(r) = -B/r.

# Possible errors:
# tolerance: 0.004
# Possible running time: < 1 s

import sys

import numpy as np
import yt
from unyt import m

tolerance = 0.0041

fn = sys.argv[1]
ds = yt.load(fn)

all_data_level_0 = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
phi = all_data_level_0["boxlib", "phi"].v.squeeze()
Er = all_data_level_0["boxlib", "Er"].v.squeeze()

Dx = ds.domain_width / ds.domain_dimensions
dr = Dx[0]
rmin = ds.domain_left_edge[0]
rmax = ds.domain_right_edge[0]
nr = phi.shape[0]
r = np.linspace(rmin + dr / 2.0, rmax - dr / 2.0, nr)
B = 1.0 / np.log(0.1 / 0.5)
A = -B * np.log(0.5)

err = 0.0
errmax_phi = 0.0
errmax_Er = 0.0
for i in range(len(r)):
    # outside EB and last cutcell
    if r[i] > 0.1 * m + dr:
        phi_theory = A + B * np.log(r[i])
        Er_theory = -B / float(r[i])
        err = abs(phi_theory - phi[i, :]).max() / phi_theory
        if err > errmax_phi:
            errmax_phi = err
        err = abs(Er_theory - Er[i, :]).max() / Er_theory
        # Exclude the last inaccurate interpolation.
        if err > errmax_Er and i < len(r) - 1:
            errmax_Er = err

print("max error of phi = ", errmax_phi)
print("max error of Er = ", errmax_Er)
print("tolerance = ", tolerance)
assert errmax_phi < tolerance and errmax_Er < tolerance
