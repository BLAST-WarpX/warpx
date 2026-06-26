#! /usr/bin/env python

# Copyright 2021 David Grote
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
This script tests the absorption of fields in the PML in RZ geometry.

The input file inputs_particle_rz is used: it features an electron
moving radially that launches a pulse. This scripts runs until
most of the pulse escapes the radial boundary. If the PML fails,
the pulse will remain with in the domain.
"""

import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(0)

# Open plotfile specified in command line
filename = sys.argv[1]
ds = yt.load(filename)

# yt 4.0+ has rounding issues with our domain data:
# RuntimeError: yt attempted to read outside the boundaries
#               of a non-periodic domain along dimension 0.
if "force_periodicity" in dir(ds):
    ds.force_periodicity()

# Check that the field is low enough
ad0 = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
Ex_array = ad0["boxlib", "Er"].to_ndarray()
Ez_array = ad0["boxlib", "Ez"].to_ndarray()
max_Ex = np.abs(Ex_array).max()
max_Ez = np.abs(Ez_array).max()
print(f"max Ex = {max_Ex}")
print(f"max Ez = {max_Ez}")
max_Efield = max(max_Ex, max_Ez)

# This tolerance was obtained empirically. As the simulation progresses, the field energy is leaking
# out through PML so that the max field diminishes with time. When the PML is working properly,
# the field level falls below 2 at the end of the simulation.
tolerance_abs = 2.0
print("tolerance_abs: " + str(tolerance_abs))
assert max_Efield < tolerance_abs
