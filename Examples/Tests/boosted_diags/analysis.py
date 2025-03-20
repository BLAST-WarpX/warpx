#!/usr/bin/env python3

# Copyright 2019 Maxence Thevenet, Revathi Jambunathan
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


"""
Analysis script of a WarpX simulation in a boosted frame.

The simulation runs in a boosted frame, and the analysis is done in the lab
frame, i.e., on the back-transformed diagnostics for the full 3D simulation and
an x-z slice at y=y_center. The field-data, Ez, along z, at (x_center,y_center,:) is compared
between the full back-transformed diagnostic and the reduced diagnostic (i.e., x-z slice) .
"""

import sys

import numpy as np
import openpmd_api as io
import yt
from openpmd_viewer import OpenPMDTimeSeries

yt.funcs.mylog.setLevel(0)

filename = sys.argv[1]

# Tolerances to check consistency between legacy BTD and new BTD
rtol = 1e-16
atol = 1e-16

# Read data from new back-transformed diagnostics (plotfile)
ds_plotfile = yt.load(filename)
data = ds_plotfile.covering_grid(
    level=0, left_edge=ds_plotfile.domain_left_edge, dims=ds_plotfile.domain_dimensions
)
Ez_plotfile = data[("mesh", "Ez")].to_ndarray()

# Read data from new back-transformed diagnostics (openPMD)
series = io.Series("./diags/diag2/openpmd_%T.h5", io.Access.read_only)
ds_openpmd = series.iterations[3]
Ez_openpmd = ds_openpmd.meshes["E"]["z"].load_chunk()
Ez_openpmd = Ez_openpmd.transpose()
series.flush()
# Compare arrays to check consistency between new BTD formats (plotfile and openPMD)
assert np.allclose(Ez_plotfile, Ez_openpmd, rtol=rtol, atol=atol)

# Check that particle random sub-selection has been applied
ts = OpenPMDTimeSeries("./diags/diag2/")
(w,) = ts.get_particle(["w"], species="beam", iteration=3)
assert (400 < len(w)) & (len(w) < 600)
