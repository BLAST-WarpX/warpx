#!/usr/bin/env python3

import os
import re
import sys

sys.path.insert(1, "../../../../warpx/Regression/Checksum/")
from checksumAPI import evaluate_checksum

test_name = os.path.split(os.getcwd())[1]
output_file = sys.argv[1]

# Run checksum regression test
if re.search("single_precision", output_file):
    evaluate_checksum(
        test_name=test_name,
        output_file=output_file,
        output_format="openpmd",
        rtol=2e-6,
    )
else:
    evaluate_checksum(
        test_name=test_name,
        output_file=output_file,
        output_format="openpmd",
    )
