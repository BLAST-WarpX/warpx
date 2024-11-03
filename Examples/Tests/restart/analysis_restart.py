#!/usr/bin/env python3

import os
import sys

sys.path.insert(1, "../../../../warpx/Regression/Checksum/")
from checksumAPI import evaluate_checksum

filename = sys.argv[1]

# Check restart data v. original data
sys.path.insert(0, "../../../../warpx/Examples/")
from analysis_default_restart import check_restart

check_restart(filename)

# compare checksums
evaluate_checksum(
    test_name=os.path.split(os.getcwd())[1],
    output_file=sys.argv[1],
)
