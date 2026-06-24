#!/usr/bin/env python

# Copyright 2024 WarpX contributors
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This script tests the embedded boundary in 3D with mesh refinement.
# A spherical surface (r=0.1) has a fixed potential 1 V.
# The outer surface has 0 V fixed.
# The analytical solution is:
# phi(r) = A + B/r, with A and B determined by boundary conditions.

import sys

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

tolerance = 0.02
print(f"tolerance = {tolerance}")

fn = sys.argv[1]

# Two concentric spherical EBs: inner at phi=1, outer at phi=0
# Analytical solution: phi(r) = A + B/r
R_inner = 0.1
R_outer = 0.5
B = 1.0 / (1.0 / R_inner - 1.0 / R_outer)
A = -B / R_outer


def get_fields(ts, level):
    if level == 0:
        phi, info = ts.get_field("phi", iteration=0)
    else:
        phi, info = ts.get_field(f"phi_lvl{level}", iteration=0)
    return phi, info


def get_error_per_lev(ts, level):
    phi, info = get_fields(ts, level)

    xx, yy, zz = np.meshgrid(info.x, info.y, info.z, indexing="ij")
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    dx = info.dx

    phi_theory = A + B / r

    # Only compare cells well away from both EB surfaces.
    # For refined levels, also exclude cells outside the fine patch
    # (openPMD reports these as zero on the full-domain fine grid).
    mask = (r > R_inner + 2 * dx) & (r < R_outer - 2 * dx)
    if level > 0:
        mask = mask & (phi != 0.0)

    phi_error = np.max(np.abs(phi[mask] - phi_theory[mask]) / np.abs(phi_theory[mask]))
    print(f"max error of phi[lev={level}]: {phi_error}")
    assert phi_error < tolerance


ts = OpenPMDTimeSeries(fn)
level_fields = [field for field in ts.avail_fields if "lvl" in field]
nlevels = 0 if level_fields == [] else int(level_fields[-1][-1])
for level in range(nlevels + 1):
    get_error_per_lev(ts, level)
