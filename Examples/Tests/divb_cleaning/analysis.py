#!/usr/bin/env python3

# Copyright 2019
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(50)

from scipy.constants import c

# Name of the last plotfile
fn = sys.argv[1]

# Load yt data
ds_old = yt.load("diags/diag1000398")
ds_mid = yt.load("diags/diag1000399")
ds_new = yt.load(fn)  # this is the last plotfile

ad_old = ds_old.covering_grid(
    level=0, left_edge=ds_old.domain_left_edge, dims=ds_old.domain_dimensions
)
ad_mid = ds_mid.covering_grid(
    level=0, left_edge=ds_mid.domain_left_edge, dims=ds_mid.domain_dimensions
)
ad_new = ds_new.covering_grid(
    level=0, left_edge=ds_new.domain_left_edge, dims=ds_new.domain_dimensions
)

G_old = ad_old["boxlib", "G"].v.squeeze()
G_new = ad_new["boxlib", "G"].v.squeeze()
divB = ad_mid["boxlib", "divB"].v.squeeze()

# Check max norm of error on c2 * div(B) = dG/dt
# (the time interval between old and new is 2*dt)
dt = 1.504557189e-15
x = G_new - G_old
y = divB * 2 * dt * c**2

rel_error = np.amax(abs(x - y)) / np.amax(abs(y))
tolerance = 1e-1

assert rel_error < tolerance
