#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
Check that velocity coincidence thinning on a cartesian velocity grid only
merged particles with the same velocity: two cold counter-propagating beams
with ux = uy = 0 must still have ux = uy = 0 and uz = +/- u0 after
resampling, and the number of macroparticles must have decreased.
"""

import sys

import numpy as np
import yt
from scipy.constants import c, m_p

yt.funcs.mylog.setLevel(0)

u0 = 0.01  # same as in the input file

ds = yt.load(sys.argv[1])
ad = ds.all_data()
w = ad["hydrogen", "particle_weight"].v
ux = ad["hydrogen", "particle_momentum_x"].v / (m_p * c)
uy = ad["hydrogen", "particle_momentum_y"].v / (m_p * c)
uz = ad["hydrogen", "particle_momentum_z"].v / (m_p * c)

n_initial = 256 * 100
print("macroparticles:", n_initial, "->", w.size)
print("max |ux|, |uy|:", np.max(np.abs(ux)), np.max(np.abs(uy)))
print("max ||uz| - u0| / u0:", np.max(np.abs(np.abs(uz) - u0)) / u0)

assert w.size < n_initial
# Merging particles with identical velocities leaves a round-off level
# transverse velocity (~1e-8 u0); mixing the two beams gives |ux| ~ u0.
assert np.max(np.abs(ux)) < 1e-6 * u0
assert np.max(np.abs(uy)) < 1e-6 * u0
assert np.allclose(np.abs(uz), u0, rtol=1e-10, atol=0.0)
# Both beams keep the same total weight
assert np.isclose(np.sum(w[uz > 0]), np.sum(w[uz < 0]), rtol=1e-10)
