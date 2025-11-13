#! /usr/bin/env python3

"""
Copyright 2020

This file is part of WarpX.

License: BSD-3-Clause-LBNL
"""

import argparse
import glob
import os
import sys

from benchmark import Benchmark
from checksum import Checksum

"""
API for WarpX checksum tests. It can be used in two ways:

- Directly use functions below to make a checksum test from a python script.
  Example: add these lines to a WarpX CI Python analysis script to run the
           checksum test.
    > import checksumAPI
    > checksumAPI.evaluate_checksum(test_name, output_file, output_format)

- As a script, to evaluate or to reset a benchmark:
  * Evaluate a benchmark. From a bash terminal:
    $ ./checksumAPI.py --evaluate --output-file <path/to/output_file> \
                       --output-format 'plotfile' --test-name <test name>
  * Reset a benchmark. From a bash terminal:
    $ ./checksumAPI.py --reset-benchmark --output-file <path/to/output_file> \
                       --output-format 'openpmd' --test-name <test name>
"""


def evaluate_checksum(
    test_name,
    output_file,
    output_format="plotfile",
    rtol=1.0e-9,
    atol=1.0e-40,
    do_fields=True,
    do_particles=True,
):
    """
    Compare output file checksum with benchmark.
    Read checksum from output file, read benchmark
    corresponding to test_name, and assert their equality.

    If the environment variable CHECKSUM_RESET is set while this function is run,
    the evaluation will be replaced with a call to reset_benchmark (see below).

    Parameters
    ----------
    test_name: string
        Name of test, as found between [] in .ini file.

    output_file : string
        Output file from which the checksum is computed.

    output_format : string
        Format of the output file (plotfile, openpmd).

    rtol: float, default=1.e-9
        Relative tolerance for the comparison.

    atol: float, default=1.e-40
        Absolute tolerance for the comparison.

    do_fields: bool, default=True
        Whether to compare fields in the checksum.

    do_particles: bool, default=True
        Whether to compare particles in the checksum.
    """
    # Reset benchmark?
    reset = os.getenv("CHECKSUM_RESET", "False").lower() in [
        "true",
        "1",
        "t",
        "y",
        "yes",
        "on",
    ]

    if reset:
        print(
            f"Environment variable CHECKSUM_RESET is set, resetting benchmark for {test_name}"
        )
        reset_benchmark(test_name, output_file, output_format, do_fields, do_particles)
    else:
        test_checksum = Checksum(
            test_name,
            output_file,
            output_format,
            do_fields=do_fields,
            do_particles=do_particles,
        )
        test_checksum.evaluate(rtol=rtol, atol=atol)


def reset_benchmark(
    test_name, output_file, output_format="plotfile", do_fields=True, do_particles=True
):
    """
    Update the benchmark (overwrites reference json file).
    Overwrite value of benchmark corresponding to
    test_name with checksum read from output file.

    Parameters
    ----------
    test_name: string
        Name of test, as found between [] in .ini file.

    output_file: string
        Output file from which the checksum is computed.

    output_format: string
        Format of the output file (plotfile, openpmd).

    do_fields: bool, default=True
        Whether to write field checksums in the benchmark.

    do_particles: bool, default=True
        Whether to write particles checksums in the benchmark.
    """
    ref_checksum = Checksum(
        test_name,
        output_file,
        output_format,
        do_fields=do_fields,
        do_particles=do_particles,
    )
    ref_benchmark = Benchmark(test_name, ref_checksum.data)
    ref_benchmark.reset()


def reset_all_benchmarks(path_to_all_output_files, output_format="plotfile"):
    """
    Update all benchmarks (overwrites reference json files)
    found in path_to_all_output_files

    Parameters
    ----------
    path_to_all_output_files: string
        Path to all output files for which the benchmarks
        are to be reset. The output files should be named <test_name>_plt, which is
        what regression_testing.regtests.py does, provided we're careful enough.

    output_format: string
        Format of the output files (plotfile, openpmd).
    """

    # Get list of output files in path_to_all_output_files
    output_file_list = glob.glob(
        path_to_all_output_files + "*_plt*[0-9]", recursive=True
    )
    output_file_list.sort()

    # Loop over output files and reset the corresponding benchmark
    for output_file in output_file_list:
        test_name = os.path.split(output_file)[1][:-9]
        reset_benchmark(test_name, output_file, output_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options relevant to evaluate a checksum or reset a benchmark
    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        default=False,
        help="Evaluate a checksum.",
    )
    parser.add_argument(
        "--reset-benchmark",
        dest="reset_benchmark",
        default=False,
        action="store_true",
        help="Reset a benchmark.",
    )
    parser.add_argument(
        "--test-name",
        dest="test_name",
        type=str,
        default="",
        required="--evaluate" in sys.argv or "--reset-benchmark" in sys.argv,
        help="Name of the test (as in WarpX-tests.ini)",
    )
    parser.add_argument(
        "--output-file",
        dest="output_file",
        type=str,
        default="",
        required="--evaluate" in sys.argv or "--reset-benchmark" in sys.argv,
        help="Name of WarpX output file",
    )
    parser.add_argument(
        "--output-format",
        dest="output_format",
        type=str,
        default="plotfile",
        required="--evaluate" in sys.argv or "--reset-benchmark" in sys.argv,
        help="Format of the output file (plotfile, openpmd)",
    )
    parser.add_argument(
        "--skip-fields",
        dest="do_fields",
        default=True,
        action="store_false",
        help="If used, do not read/write field checksums",
    )
    parser.add_argument(
        "--skip-particles",
        dest="do_particles",
        default=True,
        action="store_false",
        help="If used, do not read/write particle checksums",
    )

    # Fields and/or particles are read from output file/written to benchmark?
    parser.add_argument(
        "--rtol",
        dest="rtol",
        type=float,
        default=1.0e-9,
        help="relative tolerance for comparison",
    )
    parser.add_argument(
        "--atol",
        dest="atol",
        type=float,
        default=1.0e-40,
        help="absolute tolerance for comparison",
    )

    # Option to reset all benchmarks present in a folder.
    parser.add_argument(
        "--reset-all-benchmarks",
        dest="reset_all_benchmarks",
        action="store_true",
        default=False,
        help="Reset all benchmarks.",
    )
    parser.add_argument(
        "--path-to-all-output-files",
        dest="path_to_all_output_files",
        type=str,
        default="",
        required="--reset-all-benchmarks" in sys.argv,
        help="Directory containing all benchmark output files, \
                        typically WarpX-benchmarks generated by \
                        regression_testing/regtest.py",
    )

    args = parser.parse_args()

    if args.reset_benchmark:
        reset_benchmark(
            args.test_name,
            args.output_file,
            args.output_format,
            do_fields=args.do_fields,
            do_particles=args.do_particles,
        )

    if args.evaluate:
        evaluate_checksum(
            args.test_name,
            args.output_file,
            args.output_format,
            rtol=args.rtol,
            atol=args.atol,
            do_fields=args.do_fields,
            do_particles=args.do_particles,
        )

    if args.reset_all_benchmarks:
        if args.output_format == "openpmd":
            sys.exit("Option --reset-all-benchmarks does not work with openPMD format")
        # WARNING: this mode does not support skip-fields/particles and tolerances
        reset_all_benchmarks(args.path_to_all_output_files, args.output_format)
