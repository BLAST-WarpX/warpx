#!/usr/bin/env python3

import argparse
import os
import sys

sys.path.insert(1, "../../../../warpx/Regression/Checksum/")
from checksumAPI import evaluate_checksum

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        help="",
        type=str,
    )
    parser.add_argument(
        "--format",
        help="",
        type=str,
    )
    parser.add_argument(
        "--rtol",
        help="",
        type=float,
        required=False,
        default=1e-9,
    )
    # parse command line arguments
    args = parser.parse_args()
    # parse test name from test directory
    test_name = os.path.split(os.getcwd())[1]
    # compare checksums
    evaluate_checksum(
        test_name=test_name,
        output_file=args.path,
        output_format=args.format,
        rtol=args.rtol,
    )
