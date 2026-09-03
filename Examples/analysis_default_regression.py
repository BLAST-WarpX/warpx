#!/usr/bin/env python3

import argparse
import os
import sys

import yt

sys.path.insert(1, "../Regression/Checksum/")
from checksumAPI import evaluate_checksum
from openpmd_viewer import OpenPMDTimeSeries


def main(args):
    # parse test name from test directory
    test_name = os.path.split(os.getcwd())[1]
    if "_restart" in test_name:
        # use original test's checksums
        test_name = test_name.replace("_restart", "")
    # compare checksums
    evaluate_checksum(
        test_name=test_name,
        output_file=args.path,
        output_format=args.format,
        rtol=args.rtol,
        do_fields=args.do_fields,
        do_particles=args.do_particles,
    )


if __name__ == "__main__":
    # define parser
    parser = argparse.ArgumentParser()

    # add arguments: output path
    parser.add_argument(
        "--path",
        help="path to output file",
        type=str,
        required=True,
    )

    # add arguments: relative tolerance
    test_name = os.path.split(os.getcwd())[1]
    compute_backend = os.getenv("WARPX_COMPUTE", "").upper()
    if compute_backend in {"CUDA", "HIP", "SYCL"}:
        # GPU checksums
        default_tolerance = 1e-1
    else:
        # CPU checksums (default for restart tests is stricter)
        default_tolerance = 1e-12 if "_restart" in test_name else 1e-9
    parser.add_argument(
        "--rtol",
        help="relative tolerance to compare checksums",
        type=float,
        required=False,
        default=default_tolerance,
    )

    # add arguments: skip fields
    parser.add_argument(
        "--skip-fields",
        help="skip fields when comparing checksums",
        action="store_true",
        dest="skip_fields",
    )

    # add arguments: skip particles
    parser.add_argument(
        "--skip-particles",
        help="skip particles when comparing checksums",
        action="store_true",
        dest="skip_particles",
    )

    # parse arguments
    args = parser.parse_args()

    # set args.format automatically
    try:
        yt.load(args.path)
        args.format = "plotfile"
    except Exception:
        try:
            OpenPMDTimeSeries(args.path)
            args.format = "openpmd"
        except Exception:
            raise ValueError(f"Could not detect format for path: {args.path}") from None

    # set args.do_fields (not parsed, based on args.skip_fields)
    args.do_fields = False if args.skip_fields else True

    # set args.do_particles (not parsed, based on args.skip_particles)
    args.do_particles = False if args.skip_particles else True

    # execute main function
    main(args)
