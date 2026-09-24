#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Integration test for particle splitting with remapping current.

Uses a neutral 2D Langmuir plasma. Only electrons are split so total rho
changes locally when children move (symmetric e+/e- splitting leaves rho
~ 0 and plotfile Gauss passes even without remapping J).

Checks after step 1 (do_not_push, E=0 initially, electrons-only split):

1. max|divE - rho/eps0| / max|rho/eps0|  (needs do_remapping_current = 1)
2. Electron macroparticle count x4; positrons unchanged

Note: initialize_self_fields is intentionally off (periodic MLMG fails per species).
"""

import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(50)

from analysis_utils import check_charge_conservation

fn = sys.argv[1]
ds = yt.load(fn)
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

check_charge_conservation(data)

num_data = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")
n_electrons_init = int(num_data[0, 3])
n_positrons_init = int(num_data[0, 4])
n_electrons_final = int(num_data[-1, 3])
n_positrons_final = int(num_data[-1, 4])

print("electrons macroparticles:  {} -> {}".format(n_electrons_init, n_electrons_final))
print("positrons macroparticles:  {} -> {}".format(n_positrons_init, n_positrons_final))

assert n_electrons_final > n_electrons_init, "electrons did not split"
assert n_positrons_final == n_positrons_init, "positrons should not split"

split_factor = 4
assert n_electrons_final >= split_factor * n_electrons_init * 0.9
