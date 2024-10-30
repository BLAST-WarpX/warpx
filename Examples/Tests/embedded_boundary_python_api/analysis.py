#!/usr/bin/env python3

# This script just checks that the PICMI file executed successfully.
# If it did there will be a plotfile for the final step.

import os
import sys

sys.path.insert(1, "../../../../warpx/Regression/Checksum/")
from checksumAPI import evaluate_checksum

step = int(sys.argv[1][-5:])
assert step == 2

# compare checksums
evaluate_checksum(
    test_name=os.path.split(os.getcwd())[1],
    output_file=sys.argv[1],
)
