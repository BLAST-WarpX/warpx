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

    # Separate grid fields from particle fields. Particle fields use the
    # species name as field type; grid fields use 'boxlib'.
    particle_species = set()
    grid_fields = []
    for field in ds_benchmark.field_list:
        ftype, fname = field
        if ftype == "boxlib":
            grid_fields.append(field)
        elif ftype != "all":
            particle_species.add(ftype)

    print(f"\ntolerance = {tolerance}")
    print()

    # Compare grid fields directly (order is deterministic)
    for field in grid_fields:
        dr = ad_restart[field].squeeze().v
        db = ad_benchmark[field].squeeze().v
        error = np.amax(np.abs(dr - db))
        if np.amax(np.abs(db)) != 0.0:
            error /= np.amax(np.abs(db))
        print(f"field: {field}; error = {error}")
        assert error < tolerance

    # Compare particle fields sorted by (particle_cpu, particle_id), since
    # Redistribute() after checkpoint-restart may reorder particles across
    # tiles/ranks. The (cpu, id) pair is the unique particle key in AMReX.
    for species in sorted(particle_species):
        species_fields = [f for f in ds_benchmark.field_list if f[0] == species]

        id_r = np.atleast_1d(ad_restart[(species, "particle_id")].squeeze().v)
        id_b = np.atleast_1d(ad_benchmark[(species, "particle_id")].squeeze().v)
        cpu_r = np.atleast_1d(ad_restart[(species, "particle_cpu")].squeeze().v)
        cpu_b = np.atleast_1d(ad_benchmark[(species, "particle_cpu")].squeeze().v)

        sort_r = np.lexsort((id_r, cpu_r))
        sort_b = np.lexsort((id_b, cpu_b))

        for field in species_fields:
            if field[1] in ("particle_id", "particle_cpu"):
                continue
            dr = np.atleast_1d(ad_restart[field].squeeze().v)[sort_r]
            db = np.atleast_1d(ad_benchmark[field].squeeze().v)[sort_b]
            error = np.amax(np.abs(dr - db))
            if np.amax(np.abs(db)) != 0.0:
                error /= np.amax(np.abs(db))
            print(f"field: {field}; error = {error}")
            assert error < tolerance
    print()


# compare restart results against original results
output_file = sys.argv[1]
check_restart(output_file)
