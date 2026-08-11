#!/usr/bin/env python3
#
# --- Analysis script for the 3D hybrid-PIC cylinder compression test: checks the
# --- Python-registered field "magnetic_pressure_Bz" against Bz^2 / (2 mu0)
# --- computed from Bz written to the same dump.
#
# Like all fields, both are averaged to cell centers on output (here over the 8
# nodes of each cell, since the grid is collocated), so the dump holds
# <Bz^2>/(2 mu0) and <Bz>. They differ by the variance of Bz over those nodes:
#     p_B - <Bz>^2/(2 mu0) = Var(Bz)/(2 mu0) >= 0.
# Relative to p_B, this is small where Bz varies little across a cell compared
# to its size: in the vacuum between the plasma edge and the wall, but not inside
# the plasma, where Bz -> 0.

import glob

import numpy as np
import yt

from pywarpx import picmi

mu0 = picmi.constants.mu0

# Parameters as in inputs_test_3d_ohm_solver_cylinder_compression_picmi.py
R_c = 0.5  # flux conserver (wall) radius (m)
R_p = 0.25  # plasma radius (m)
delta_p = 0.025  # plasma edge width (m)

dumps = sorted(glob.glob("diags/diag1??????"))
assert len(dumps) == 2, f"expected the step-0 and final dumps, found {dumps}"

for dump in dumps:
    ds = yt.load(dump)
    ad = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    Bz = ad["boxlib", "Bz"].v
    p_B = ad["boxlib", "magnetic_pressure_Bz"].v
    r = np.hypot(ad["boxlib", "x"].v, ad["boxlib", "y"].v)
    dx = (ds.domain_width / ds.domain_dimensions).v[0]

    p_from_Bz = Bz**2 / (2.0 * mu0)

    # (1) p_out >= <Bz>^2/(2 mu0) everywhere, up to round-off
    min_diff = np.min(p_B - p_from_Bz)
    print(
        f"{dump}: max p_B = {p_B.max():.3f} Pa, min(p_B - Bz^2/2mu0) = {min_diff:.3e} Pa"
    )
    assert min_diff >= -1e-12 * p_B.max()

    # (2) agreement in the vacuum region R_p + 6 delta_p <= r <= R_c - dx.
    # There Bz ~ B0 (1 - f/2) with f = exp(-(r - R_p)/delta_p) <= exp(-6), so Bz
    # changes across a cell by about (f/2)(dx/delta_p) Bz, and the relative
    # difference is of order (f dx / (4 delta_p))^2 (~7e-7 at this resolution).
    # This leading-order estimate ignores the growth of f within the cell and
    # the plasma response, so allow a factor of 20.
    vacuum = (r >= R_p + 6.0 * delta_p) & (r <= R_c - dx)
    rel_diff = np.max(np.abs(p_B[vacuum] / p_from_Bz[vacuum] - 1.0))
    estimate = (np.exp(-6.0) * dx / (4.0 * delta_p)) ** 2
    print(
        f"  vacuum region: max rel. difference = {rel_diff:.2e} ({rel_diff / estimate:.1f} x estimate)"
    )
    assert rel_diff < 20.0 * estimate
