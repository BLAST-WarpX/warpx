#!/usr/bin/env python3

# Copyright 2019-2020 Luca Fedeli, Maxence Thevenet
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


"""
Analysis script of a WarpX simulation of rigid injection.

A Gaussian electron beam starts from -5 microns, propagates rigidly up to
20 microns after which it expands due to emittance only (the focal position is
20 microns). The beam width is measured after ~50 microns, and compared with
the theory (with a 5% error allowed).

As a help to the user, the script also compares beam width to the theory in
case rigid injection is OFF (i.e., the beam starts expanding from -5 microns),
in which case a warning is raised.

Additionally, this script tests that runtime attributes are correctly initialized
with the gaussian_beam injection style.
"""

import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(0)

filename = sys.argv[1]


# WarpX headers include more data when rigid injection is used,
# which gives an error with the last yt release.
# To avoid this issue, the three last lines of WarpXHeader are removed if
# needed.
def remove_rigid_lines(plotfile, nlines_if_rigid):
    header_name = plotfile + "/WarpXHeader"
    f = open(header_name, "r")
    file_lines = f.readlines()
    nlines = len(file_lines)
    f.close()
    if nlines == nlines_if_rigid:
        f = open(header_name, "w")
        f.writelines(file_lines[:-3])
        f.close()


# Remove rigid injection header lines
remove_rigid_lines(filename, 18)
# Read beam parameters
ds = yt.load(filename)
ad = ds.all_data()
# Beam longitudinal position
z = np.mean(ad["beam", "particle_position_y"].v)
# Beam width
w = np.std(ad["beam", "particle_position_x"].v)

# initial parameters
z0 = 20.0e-6
z0_no_rigid = -5.0e-6
w0 = 1.0e-6
theta0 = np.arcsin(0.1)

# Theoretical beam width after propagation if rigid OFF
# Inform the user if rigid injection simply off (just to be kind)
wth_no_rigid = np.sqrt(w0**2 + (z - z0_no_rigid) ** 2 * theta0**2)
error_no_rigid = np.abs((w - wth_no_rigid) / wth_no_rigid)
if error_no_rigid < 0.05:
    print("error no rigid: " + str(error_no_rigid))
    print("Looks like the beam defocuses as if rigid injection were OFF")

# Theoretical beam width after propagation if rigid ON
wth = np.sqrt(w0**2 + (z - z0) ** 2 * theta0**2)
error_rel = np.abs((w - wth) / wth)
tolerance_rel = 0.05

# Print error and assert small error
print("Beam position: " + str(z))
print("Beam width   : " + str(w))

print("error_rel    : " + str(error_rel))
print("tolerance_rel: " + str(tolerance_rel))

assert error_rel < tolerance_rel


### Check that user runtime attributes are correctly initialized
filename_start = filename[:-5] + "00000"
ds_start = yt.load(filename_start)
ad_start = ds_start.all_data()
x = ad_start["beam", "particle_position_x"]
z = ad_start["beam", "particle_position_y"]
orig_z = ad_start["beam", "particle_orig_z"]
center = ad_start["beam", "particle_center"]
assert np.array_equal(z, orig_z)
assert np.array_equal(1 * (np.abs(x) < 5.0e-7), center)
