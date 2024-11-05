#!/usr/bin/env python3

import argparse
import os
import sys

sys.path.insert(1, "../../../../warpx/Regression/Checksum/")
from checksumAPI import evaluate_checksum


def main(args):
    # parse test name from test director (remove "_restart" suffix
    # for restart tests, same checksums as original test)
    test_name = os.path.split(os.getcwd())[1]
    test_name = test_name.replace("_restart", "")
    # compare checksums
    evaluate_checksum(
        test_name=test_name,
        output_file=args.path,
        output_format=args.format,
        rtol=args.rtol,
        do_particles=args.do_particles,
    )


if __name__ == "__main__":
    # define parser
    parser = argparse.ArgumentParser()
    # add arguments: output path
    parser.add_argument(
        "--path",
        help="",
        type=str,
    )
    # add arguments: output format
    format_group = parser.add_mutually_exclusive_group(required=True)
    format_group.add_argument(
        "--plotfile",
        help="",
        action="store_true",
    )
    format_group.add_argument(
        "--openpmd",
        help="",
        action="store_true",
    )
    # add arguments: relative tolerance
    parser.add_argument(
        "--rtol",
        help="",
        type=float,
        required=False,
        default=1e-9,
    )
    parser.add_argument(
        "--skip-particles",
        help="",
        action="store_true",
        dest="skip_particles",
    )
    # parse arguments
    args = parser.parse_args()
    # set args.format (based on args.plotfile and args.openpmd)
    args.format = "plotfile" if args.plotfile else "openpmd"
    # set args.do_particles (based on args.skip_particles)
    args.do_particles = False if args.skip_particles else True
    # TODO check environment and reset tolerance (portable, machine precision)
    # execute main function
    main(args)
