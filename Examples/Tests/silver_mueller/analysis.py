#!/usr/bin/env python3

# Copyright 2021
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
"""
This tests verifies the Silver-Mueller boundary condition.
A laser pulse is emitted and propagates towards the boundaries ; the
test check that the reflected field at the boundary is negligible.
"""

import sys

import numpy as np
import yt

sys.path.append("../../../Tools/Parser/")
from input_file_parser import input_has_value, parse_input_file

yt.funcs.mylog.setLevel(0)

filename = sys.argv[1]

ds = yt.load(filename)
all_data_level_0 = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
input_dict = parse_input_file("./warpx_used_inputs")
geom_RZ = input_has_value(input_dict, "geometry.dims", "RZ")
if geom_RZ:
    Er = all_data_level_0["boxlib", "Er"].v.squeeze()
    Et = all_data_level_0["boxlib", "Et"].v.squeeze()
    Ez = all_data_level_0["boxlib", "Ez"].v.squeeze()
else:
    Ex = all_data_level_0["boxlib", "Ex"].v.squeeze()
    Ey = all_data_level_0["boxlib", "Ey"].v.squeeze()
    Ez = all_data_level_0["boxlib", "Ez"].v.squeeze()
# The peak of the initial laser pulse is on the order of 6 V/m
# Check that the amplitude after reflection is less than 0.01 V/m
max_reflection_amplitude = 0.01

if geom_RZ:
    assert np.all(abs(Er) < max_reflection_amplitude)
    assert np.all(abs(Et) < max_reflection_amplitude)
    assert np.all(abs(Ez) < max_reflection_amplitude)
else:
    assert np.all(abs(Ex) < max_reflection_amplitude)
    assert np.all(abs(Ey) < max_reflection_amplitude)
    assert np.all(abs(Ez) < max_reflection_amplitude)
