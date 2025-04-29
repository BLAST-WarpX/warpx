#!/usr/bin/env python3

import os
import sys

import numpy as np
import yt


def check_restart(filename, tolerance=1e-12):
    """
    Compare output data generated from initial run with output data generated after restart.

    Parameters
    ----------
    filename : str
        Name of the plotfile containing the output data generated after restart.
    tolerance : float, optional (default = 1e-12)
        Relative error between restart and original data must be smaller than tolerance.
    """
    # Load output data generated after restart
    ds_restart = yt.load(filename)

    # yt 4.0+ has rounding issues with our domain data:
    # RuntimeError: yt attempted to read outside the boundaries
    # of a non-periodic domain along dimension 0.
    if "force_periodicity" in dir(ds_restart):
        ds_restart.force_periodicity()

    ad_restart = ds_restart.covering_grid(
        level=0,
        left_edge=ds_restart.domain_left_edge,
        dims=ds_restart.domain_dimensions,
    )

    # Load output data generated from initial run
    benchmark = os.path.join(os.getcwd().replace("_restart", ""), filename)
    ds_benchmark = yt.load(benchmark)

    # yt 4.0+ has rounding issues with our domain data:
    # RuntimeError: yt attempted to read outside the boundaries
    # of a non-periodic domain along dimension 0.
    if "force_periodicity" in dir(ds_benchmark):
        ds_benchmark.force_periodicity()

    ad_benchmark = ds_benchmark.covering_grid(
        level=0,
        left_edge=ds_benchmark.domain_left_edge,
        dims=ds_benchmark.domain_dimensions,
    )

    # Loop over all fields (all particle species, all particle attributes, all grid fields)
    # and compare output data generated from initial run with output data generated after restart
    print(f"\ntolerance = {tolerance}")
    print()
    for field in ds_benchmark.field_list:
        dr = ad_restart[field].squeeze().v
        db = ad_benchmark[field].squeeze().v
        error = np.amax(np.abs(dr - db))
        if np.amax(np.abs(db)) != 0.0:
            error /= np.amax(np.abs(db))
        print(f"field: {field}; error = {error}")
        assert error < tolerance
    print()


# compare restart results against original results
output_file = sys.argv[1]
check_restart(output_file)
