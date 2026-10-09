#!/usr/bin/env python3

"""
Check that moving-window restarts preserve diagnostic data and domain bounds.

This complements analysis_default_restart.py by comparing both the number of
cells and the physical extent. Incorrect bounds can change the recorded cell
size even when the number of cells and field values are unchanged.
"""

import argparse
import os

import numpy as np
import yt
from analysis_default_restart import check_restart


def check_restart_domain(filename, tolerance=1e-12):
    """
    Compare the domain bounds of a restarted run with those of the
    corresponding direct (non-restarted) run.

    Parameters
    ----------
    filename : str
        Name of the plotfile containing the output data generated after restart.
    tolerance : float, optional (default = 1e-12)
        Relative error between restart and original domain bounds must be smaller
        than tolerance.
    """
    ds_restart = yt.load(filename)

    benchmark = os.path.join(os.getcwd().replace("_restart", ""), filename)
    ds_benchmark = yt.load(benchmark)

    np.testing.assert_array_equal(
        ds_restart.domain_dimensions, ds_benchmark.domain_dimensions
    )

    left_restart = ds_restart.domain_left_edge.v
    right_restart = ds_restart.domain_right_edge.v
    left_benchmark = ds_benchmark.domain_left_edge.v
    right_benchmark = ds_benchmark.domain_right_edge.v

    print(f"\ntolerance = {tolerance}")
    print(
        f"restart:   domain_left_edge = {left_restart}, domain_right_edge = {right_restart}"
    )
    print(
        f"benchmark: domain_left_edge = {left_benchmark}, domain_right_edge = {right_benchmark}"
    )

    scale = np.amax(np.abs(right_benchmark - left_benchmark))
    error_lo = np.amax(np.abs(left_restart - left_benchmark)) / scale
    error_hi = np.amax(np.abs(right_restart - right_benchmark)) / scale
    print(f"error_lo = {error_lo}, error_hi = {error_hi}")
    assert error_lo < tolerance
    assert error_hi < tolerance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        help="path to output file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--rtol",
        help="relative tolerance between restart and original",
        type=float,
        required=False,
        default=1e-12,
    )
    args = parser.parse_args()

    check_restart_domain(filename=args.path, tolerance=args.rtol)
    check_restart(filename=args.path, tolerance=args.rtol)
