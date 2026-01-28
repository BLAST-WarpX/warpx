#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Andrew Myers
# License: BSD-3-Clause-LBNL
#
# This script plots the fields from the 2D particle absorbing boundary test case.


import yt

ds = yt.load("diags/diagInst001000/")

field = "Ez"

# This is to swap the SlicePlot from vertical to horizontal
ds.coordinates.x_axis["z"] = 1
ds.coordinates.x_axis[2] = 1
ds.coordinates.y_axis["z"] = 0
ds.coordinates.y_axis[2] = 0

sl = yt.SlicePlot(ds, "z", field, aspect=100, origin='native')
sl.set_zlim(field, -1.e7, 1.e7)
sl.set_log(field, log=False)
sl.set_ylabel(r"x  $\mathrm{(\mu m)}$")
sl.set_xlabel(r"z  $\mathrm{(\mu m)}$")
#sl.annotate_grids()

sl.save("particle_absorbing_boundary_2d_"+field+".png")