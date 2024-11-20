#!/usr/bin/env python3

import os
import sys

import numpy as np

sys.path.insert(1, "../../../../warpx/Regression/Checksum/")
from checksumAPI import evaluate_checksum

# check that the maximum chi value is small
fname = "diags/reducedfiles/ParticleExtrema_beam_p.txt"
chi_max = np.loadtxt(fname)[:, 19]
assert np.all(chi_max < 2e-8)

# check that no photons have been produced
fname = "diags/reducedfiles/ParticleNumber.txt"
pho_num = np.loadtxt(fname)[:, 7]
assert pho_num.all() == 0.0

# compare checksums
evaluate_checksum(
    test_name=os.path.split(os.getcwd())[1],
    output_file=sys.argv[1],
)
