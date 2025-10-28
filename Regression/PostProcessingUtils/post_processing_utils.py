# Copyright 2021-2021 Neil Zaim
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

## This file contains functions that are used in multiple CI analysis scripts.

import numpy as np
import yt


## This is a generic function to test a particle filter. We reproduce the filter in python and
## verify that the results are the same as with the WarpX filtered diagnostic.
def check_particle_filter(fn, filtered_fn, filter_expression, dim, species_name):
    ds = yt.load(fn)
    ds_filtered = yt.load(filtered_fn)
    ad = ds.all_data()
    ad_filtered = ds_filtered.all_data()

    ## Load arrays from the unfiltered diagnostic
    ids = ad[species_name, "particle_id"].to_ndarray()
    cpus = ad[species_name, "particle_cpu"].to_ndarray()
    px = ad[species_name, "particle_momentum_x"].to_ndarray()
    pz = ad[species_name, "particle_momentum_z"].to_ndarray()
    py = ad[species_name, "particle_momentum_y"].to_ndarray()
    w = ad[species_name, "particle_weight"].to_ndarray()
    if dim == "2d":
        x = ad[species_name, "particle_position_x"].to_ndarray()
        z = ad[species_name, "particle_position_y"].to_ndarray()
    elif dim == "3d":
        x = ad[species_name, "particle_position_x"].to_ndarray()
        y = ad[species_name, "particle_position_y"].to_ndarray()
        z = ad[species_name, "particle_position_z"].to_ndarray()
    elif dim == "rz":
        r = ad[species_name, "particle_position_x"].to_ndarray()
        z = ad[species_name, "particle_position_y"].to_ndarray()
        theta = ad[species_name, "particle_theta"].to_ndarray()

    ## Load arrays from the filtered diagnostic
    ids_filtered_warpx = ad_filtered[species_name, "particle_id"].to_ndarray()
    cpus_filtered_warpx = ad_filtered[species_name, "particle_cpu"].to_ndarray()
    px_filtered_warpx = ad_filtered[species_name, "particle_momentum_x"].to_ndarray()
    pz_filtered_warpx = ad_filtered[species_name, "particle_momentum_z"].to_ndarray()
    py_filtered_warpx = ad_filtered[species_name, "particle_momentum_y"].to_ndarray()
    w_filtered_warpx = ad_filtered[species_name, "particle_weight"].to_ndarray()
    if dim == "2d":
        x_filtered_warpx = ad_filtered[species_name, "particle_position_x"].to_ndarray()
        z_filtered_warpx = ad_filtered[species_name, "particle_position_y"].to_ndarray()
    elif dim == "3d":
        x_filtered_warpx = ad_filtered[species_name, "particle_position_x"].to_ndarray()
        y_filtered_warpx = ad_filtered[species_name, "particle_position_y"].to_ndarray()
        z_filtered_warpx = ad_filtered[species_name, "particle_position_z"].to_ndarray()
    elif dim == "rz":
        r_filtered_warpx = ad_filtered[species_name, "particle_position_x"].to_ndarray()
        z_filtered_warpx = ad_filtered[species_name, "particle_position_y"].to_ndarray()
        theta_filtered_warpx = ad_filtered[species_name, "particle_theta"].to_ndarray()

    ## Reproduce the filter in python: this returns the indices of the filtered particles in the
    ## unfiltered arrays.
    (ind_filtered_python,) = np.where(eval(filter_expression))

    ## Sort the indices of the filtered arrays by particle id.
    sorted_ind_filtered_python = ind_filtered_python[
        np.argsort(ids[ind_filtered_python])
    ]
    sorted_ind_filtered_warpx = np.argsort(ids_filtered_warpx)

    ## Check that the sorted ids are exactly the same with the warpx filter and the filter
    ## reproduced in python
    assert np.array_equal(
        ids[sorted_ind_filtered_python], ids_filtered_warpx[sorted_ind_filtered_warpx]
    )
    assert np.array_equal(
        cpus[sorted_ind_filtered_python], cpus_filtered_warpx[sorted_ind_filtered_warpx]
    )

    ## Finally, we check that the sum of the particles quantities are the same to machine precision
    tolerance_checksum = 1.0e-12
    check_array_sum(
        px[sorted_ind_filtered_python],
        px_filtered_warpx[sorted_ind_filtered_warpx],
        tolerance_checksum,
    )
    check_array_sum(
        pz[sorted_ind_filtered_python],
        pz_filtered_warpx[sorted_ind_filtered_warpx],
        tolerance_checksum,
    )
    check_array_sum(
        py[sorted_ind_filtered_python],
        py_filtered_warpx[sorted_ind_filtered_warpx],
        tolerance_checksum,
    )
    check_array_sum(
        w[sorted_ind_filtered_python],
        w_filtered_warpx[sorted_ind_filtered_warpx],
        tolerance_checksum,
    )
    check_array_sum(
        z[sorted_ind_filtered_python],
        z_filtered_warpx[sorted_ind_filtered_warpx],
        tolerance_checksum,
    )
    if dim == "2d":
        check_array_sum(
            x[sorted_ind_filtered_python],
            x_filtered_warpx[sorted_ind_filtered_warpx],
            tolerance_checksum,
        )
    elif dim == "3d":
        check_array_sum(
            x[sorted_ind_filtered_python],
            x_filtered_warpx[sorted_ind_filtered_warpx],
            tolerance_checksum,
        )
        check_array_sum(
            y[sorted_ind_filtered_python],
            y_filtered_warpx[sorted_ind_filtered_warpx],
            tolerance_checksum,
        )
    elif dim == "rz":
        check_array_sum(
            r[sorted_ind_filtered_python],
            r_filtered_warpx[sorted_ind_filtered_warpx],
            tolerance_checksum,
        )
        check_array_sum(
            theta[sorted_ind_filtered_python],
            theta_filtered_warpx[sorted_ind_filtered_warpx],
            tolerance_checksum,
        )


## This function checks that the absolute sums of two arrays are the same to a required precision
def check_array_sum(array1, array2, tolerance_checksum):
    sum1 = np.sum(np.abs(array1))
    sum2 = np.sum(np.abs(array2))
    assert abs(sum2 - sum1) / sum1 < tolerance_checksum


## This function is specifically used to test the random filter. First, we check that the number of
## dumped particles is as expected. Next, we call the generic check_particle_filter function.
def check_random_filter(fn, filtered_fn, random_fraction, dim, species_name):
    ds = yt.load(fn)
    ds_filtered = yt.load(filtered_fn)
    ad = ds.all_data()
    ad_filtered = ds_filtered.all_data()

    ## Check that the number of particles is as expected
    numparts = ad[species_name, "particle_id"].to_ndarray().shape[0]
    numparts_filtered = ad_filtered["particle_id"].to_ndarray().shape[0]
    expected_numparts_filtered = random_fraction * numparts
    # 5 sigma test that has an intrinsic probability to fail of 1 over ~2 millions
    std_numparts_filtered = np.sqrt(expected_numparts_filtered)
    error = abs(numparts_filtered - expected_numparts_filtered)
    print(
        "Random filter: difference between expected and actual number of dumped particles: "
        + str(error)
    )
    print("tolerance: " + str(5 * std_numparts_filtered))
    assert error < 5 * std_numparts_filtered

    ## Dirty trick to find particles with the same ID + same CPU (does not work with more than 10
    ## MPI ranks)
    random_filter_expression = (
        "np.isin(ids + 0.1*cpus,ids_filtered_warpx + 0.1*cpus_filtered_warpx)"
    )
    check_particle_filter(fn, filtered_fn, random_filter_expression, dim, species_name)
