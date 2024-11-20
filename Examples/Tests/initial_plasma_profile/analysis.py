#!/usr/bin/env python3

# Copyright 2020 Michael Rowan
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import os
import sys

sys.path.insert(1, "../../../../warpx/Regression/Checksum/")
from checksumAPI import evaluate_checksum

# compare checksums
evaluate_checksum(
    test_name=os.path.split(os.getcwd())[1],
    output_file=sys.argv[1],
    rtol=1e-4,
    do_particles=False,
)
