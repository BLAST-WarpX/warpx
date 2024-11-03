#!/usr/bin/env python3

# Copyright 2021 Modern Electron

# This script checks that the inputs_test_2d_background_mcc_picmi.py run more-or-less matches the
# results from the non-PICMI run. The PICMI run is using an external Poisson
# solver that directly solves the Poisson equation using matrix inversion
# rather than the iterative approach from the MLMG solver.

import os
import sys

sys.path.insert(1, "../../../../warpx/Regression/Checksum/")
from checksumAPI import evaluate_checksum

# compare checksums
evaluate_checksum(
    test_name=os.path.split(os.getcwd())[1],
    output_file=sys.argv[1],
    rtol=5e-3,
)
