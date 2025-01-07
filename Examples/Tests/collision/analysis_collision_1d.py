#!/usr/bin/env python3

# Copyright 2024 Justin Angus
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from the script `inputs_1d`.
# run locally: python analysis_vandb_1d.py diags/diag1000600/
#
# This is a 1D intra-species Coulomb scattering relaxation test consisting
# of a low-density population streaming into a higher density population at rest.
# Both populations belong to the same carbon12 ion species.
# See test T1b from JCP 413 (2020) by D. Higginson, et al.
#
import sys

import numpy as np
import yt
from scipy.constants import e

# this will be the name of the plot file
last_fn = sys.argv[1]
ds = yt.load(last_fn)
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

# carbon 12 ion (mass = 12*amu - 6*me)
mass = 1.992100316897910e-26

# Separate macroparticles from group A (low weight) and group B (high weight)
# by sorting based on weight
sorted_indices = data["ions", "particle_weight"].argsort()
sorted_wp = data["ions", "particle_weight"][sorted_indices].value
sorted_px = data["ions", "particle_momentum_x"][sorted_indices].value
sorted_py = data["ions", "particle_momentum_y"][sorted_indices].value
sorted_pz = data["ions", "particle_momentum_z"][sorted_indices].value

# Find the index 'Npmin' that separates macroparticles from group A and group B
Np = len(sorted_wp)
wpmin = sorted_wp.min()
wpmax = sorted_wp.max()
for i in range(len(sorted_wp)):
    if sorted_wp[i] > wpmin:
        Npmin = i
        break

NpA = Npmin
wpA = wpmin
NpB = Np - Npmin
wpB = wpmax
NpAs = 0
NpAe = Npmin
NpBs = Npmin
NpBe = Np

#############

sorted_px_sum = np.abs(sorted_px).sum()
sorted_py_sum = np.abs(sorted_py).sum()
sorted_pz_sum = np.abs(sorted_pz).sum()
sorted_wp_sum = np.abs(sorted_wp).sum()

# compute mean velocities
wAtot = wpA * NpA
wBtot = wpB * NpB

uBx = uBy = uBz = 0.0
for i in range(NpBs, NpBe):
    uBx += wpB * sorted_px[i]
    uBy += wpB * sorted_py[i]
    uBz += wpB * sorted_pz[i]
uBx /= mass * wBtot  # [m/s]
uBy /= mass * wBtot  # [m/s]
uBz /= mass * wBtot  # [m/s]

uAx = uAy = uAz = 0.0
for i in range(NpAs, NpAe):
    uAx += wpA * sorted_px[i]
    uAy += wpA * sorted_py[i]
    uAz += wpA * sorted_pz[i]
uAx /= mass * wAtot  # [m/s]
uAy /= mass * wAtot  # [m/s]
uAz /= mass * wAtot  # [m/s]

# compute temperatures
TBx = TBy = TBz = 0.0
for i in range(NpBs, NpBe):
    TBx += wpB * (sorted_px[i] / mass - uBx) ** 2
    TBy += wpB * (sorted_py[i] / mass - uBy) ** 2
    TBz += wpB * (sorted_pz[i] / mass - uBz) ** 2
TBx *= mass / (e * wBtot)
TBy *= mass / (e * wBtot)
TBz *= mass / (e * wBtot)

TAx = TAy = TAz = 0.0
for i in range(NpAs, NpAe):
    TAx += wpA * (sorted_px[i] / mass - uAx) ** 2
    TAy += wpA * (sorted_py[i] / mass - uAy) ** 2
    TAz += wpA * (sorted_pz[i] / mass - uAz) ** 2
TAx *= mass / (e * wAtot)
TAy *= mass / (e * wAtot)
TAz *= mass / (e * wAtot)

TApar = TAz
TAperp = (TAx + TAy) / 2.0
TA = (TAx + TAy + TAz) / 3.0

TBpar = TBz
TBperp = (TBx + TBy) / 2.0
TB = (TBx + TBy + TBz) / 3.0

TApar_30ps_soln = 6.15e3  # TA parallel solution at t = 30 ps
error = np.abs(TApar - TApar_30ps_soln) / TApar_30ps_soln
tolerance = 0.02
print("TApar at 30ps error = ", error)
print("tolerance = ", tolerance)
assert error < tolerance
