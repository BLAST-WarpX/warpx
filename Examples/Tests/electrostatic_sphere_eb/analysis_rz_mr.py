#!/usr/bin/env python

# Copyright 2024 Olga Shapoval, Edoardo Zoni
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This script tests the embedded boundary in RZ.
# A cylindrical surface (r=0.1) has a fixed potential 1 V.
# The outer surface has 0 V fixed.
# Thus the analytical solution has the form:
# phi(r) = A+B*log(r), Er(r) = -B/r.

import sys

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

tolerance = 0.004
print(f"tolerance = {tolerance}")

fn = sys.argv[1]


def find_first_non_zero_from_bottom_left(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if (matrix[i][j] != 0) and (matrix[i][j] != np.nan):
                return (i, j)
    return i, j


def find_first_non_zero_from_upper_right(matrix):
    for i in range(matrix.shape[0] - 1, -1, -1):
        for j in range(matrix.shape[1] - 1, -1, -1):
            if (matrix[i][j] != 0) and (matrix[i][j] != np.nan):
                return (i, j)
    return i, j


def get_fields(ts, level):
    if level == 0:
        Er, info = ts.get_field("E", "r", iteration=0)
        phi, info = ts.get_field("phi", iteration=0)
    else:
        Er, info = ts.get_field(f"E_lvl{level}", "r", iteration=0)
        phi, info = ts.get_field(f"phi_lvl{level}", iteration=0)
    return Er, phi, info


def get_error_per_lev(ts, level):
    Er, phi, info = get_fields(ts, level)

    nr_half = info.r.shape[0] // 2
    dr = info.dr

    Er_patch = Er[:, nr_half:]
    phi_patch = phi[:, nr_half:]
    r1 = info.r[nr_half:]
    patch_left_lower_i, patch_left_lower_j = find_first_non_zero_from_bottom_left(
        Er_patch
    )
    patch_right_upper_i, patch_right_upper_j = find_first_non_zero_from_upper_right(
        Er_patch
    )

    # phi and Er field on the MR patch
    phi_sim = phi_patch[
        patch_left_lower_i : patch_right_upper_i + 1,
        patch_left_lower_j : patch_right_upper_j + 1,
    ]
    Er_sim = Er_patch[
        patch_left_lower_i : patch_right_upper_i + 1,
        patch_left_lower_j : patch_right_upper_j + 1,
    ]
    r = r1[patch_left_lower_j : patch_right_upper_j + 1]

    B = 1.0 / np.log(0.1 / 0.5)
    A = -B * np.log(0.5)

    # outside EB and last cutcell
    rmin = np.min(np.argwhere(r >= (0.1 + dr)))
    rmax = -1
    r = r[rmin:rmax]
    phi_sim = phi_sim[:, rmin:rmax]
    Er_sim = Er_sim[:, rmin:rmax]

    phi_theory = A + B * np.log(r)
    phi_theory = np.tile(phi_theory, (phi_sim.shape[0], 1))
    phi_error = np.max(np.abs(phi_theory - phi_sim) / np.abs(phi_theory))

    Er_theory = -B / r
    Er_theory = np.tile(Er_theory, (Er_sim.shape[0], 1))
    Er_error = np.max(np.abs(Er_theory - Er_sim) / np.abs(Er_theory))

    print(f"max error of phi[lev={level}]: {phi_error}")
    print(f"max error of  Er[lev={level}]: {Er_error}")
    assert phi_error < tolerance
    assert Er_error < tolerance


ts = OpenPMDTimeSeries(fn)
level_fields = [field for field in ts.avail_fields if "lvl" in field]
nlevels = 0 if level_fields == [] else int(level_fields[-1][-1])
for level in range(nlevels + 1):
    get_error_per_lev(ts, level)
