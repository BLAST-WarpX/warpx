#!/usr/bin/env python3

import argparse
import os
import sys

import yt
from openpmd_viewer import OpenPMDTimeSeries

sys.path.insert(1, "../../../../warpx/Regression/Checksum/")
from checksumAPI import evaluate_checksum


def main(args):
    # parse test name from test directory
    test_name = os.path.split(os.getcwd())[1]
    if "_restart" in test_name:
        rtol_restart = 1e-12
        print(
            f"Warning: Setting relative tolerance {rtol_restart} for restart checksum analysis"
        )
        # use original test's checksums
        test_name = test_name.replace("_restart", "")
        # reset relative tolerance
        args.rtol = rtol_restart
    # TODO check environment and reset tolerance (portable, machine precision)
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
        help="path to output file(s)",
        type=str,
    )
    # add arguments: relative tolerance
    parser.add_argument(
        "--rtol",
        help="relative tolerance to compare checksums",
        type=float,
        required=False,
        default=1e-9,
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
    except Exception:
        try:
            OpenPMDTimeSeries(args.path)
        except Exception:
            print("Could not open the file as a plotfile or an openPMD time series")
        else:
            args.format = "openpmd"
    else:
        args.format = "plotfile"
    # set args.do_fields (not parsed, based on args.skip_fields)
    args.do_fields = False if args.skip_fields else True
    # set args.do_particles (not parsed, based on args.skip_particles)
    args.do_particles = False if args.skip_particles else True
    # execute main function
    main(args)
