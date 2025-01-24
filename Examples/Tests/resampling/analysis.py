#!/usr/bin/env python3

# Copyright 2020 Neil Zaim
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

## In this test, we check that leveling thinning works as expected on two simple cases. Each case
## corresponds to a different particle species.

import sys

import numpy as np
import yt
from scipy.special import erf

fn_final = sys.argv[1]
fn0 = fn_final[:-4] + "0000"

ds0 = yt.load(fn0)
ds = yt.load(fn_final)

ad0 = ds0.all_data()
ad = ds.all_data()

numcells = 16 * 16
t_r = 1.3  # target ratio
relative_tol = 1.0e-13  # tolerance for machine precision errors


#### Tests for first species ####
# Particles are present in all simulation cells and all have the same weight

ppc = 400.0
numparts_init = numcells * ppc

w0 = ad0["resampled_part1", "particle_weight"].to_ndarray()  # weights before resampling
w = ad["resampled_part1", "particle_weight"].to_ndarray()  # weights after resampling

# Renormalize weights for easier calculations
w0 = w0 * ppc
w = w * ppc

# Check that initial number of particles is indeed as expected
assert w0.shape[0] == numparts_init
# Check that all initial particles have the same weight
assert np.unique(w0).shape[0] == 1
# Check that this weight is 1 (to machine precision)
assert abs(w0[0] - 1) < relative_tol

# Check that the number of particles after resampling is as expected
numparts_final = w.shape[0]
expected_numparts_final = numparts_init / t_r**2
error = np.abs(numparts_final - expected_numparts_final)
std_numparts_final = np.sqrt(numparts_init / t_r**2 * (1.0 - 1.0 / t_r**2))
# 5 sigma test that has an intrinsic probability to fail of 1 over ~2 millions
print(
    "difference between expected and actual final number of particles (1st species): "
    + str(error)
)
print("tolerance: " + str(5 * std_numparts_final))
assert error < 5 * std_numparts_final

# Check that the final weight is the same for all particles (to machine precision) and is as
# expected
final_weight = t_r**2
assert np.amax(np.abs(w - final_weight)) < relative_tol * final_weight


#### Tests for second species ####
# Particles are only present in a single cell and have a gaussian weight distribution
# Using a single cell makes the analysis easier because leveling thinning is done separately in
# each cell

ppc = 100000.0
numparts_init = ppc

w0 = ad0["resampled_part2", "particle_weight"].to_ndarray()  # weights before resampling
w = ad["resampled_part2", "particle_weight"].to_ndarray()  # weights after resampling

# Renormalize and sort weights for easier calculations
w0 = np.sort(w0) * ppc
w = np.sort(w) * ppc

## First we verify that the initial distribution is as expected

# Check that the mean initial weight is as expected
mean_initial_weight = np.average(w0)
expected_mean_initial_weight = 2.0 * np.sqrt(2.0)
error = np.abs(mean_initial_weight - expected_mean_initial_weight)
expected_std_initial_weight = 1.0 / np.sqrt(2.0)
std_mean_initial_weight = expected_std_initial_weight / np.sqrt(numparts_init)
# 5 sigma test that has an intrinsic probability to fail of 1 over ~2 millions
print(
    "difference between expected and actual mean initial weight (2nd species): "
    + str(error)
)
print("tolerance: " + str(5 * std_mean_initial_weight))
assert error < 5 * std_mean_initial_weight

# Check that the initial weight variance is as expected
variance_initial_weight = np.var(w0)
expected_variance_initial_weight = 0.5
error = np.abs(variance_initial_weight - expected_variance_initial_weight)
std_variance_initial_weight = expected_variance_initial_weight * np.sqrt(
    2.0 / numparts_init
)
# 5 sigma test that has an intrinsic probability to fail of 1 over ~2 millions
print(
    "difference between expected and actual variance of initial weight (2nd species): "
    + str(error)
)
print("tolerance: " + str(5 * std_variance_initial_weight))

## Next we verify that the resampling worked as expected

# Check that the level weight value is as expected from the initial distribution
level_weight = w[0]
assert np.abs(level_weight - mean_initial_weight * t_r) < level_weight * relative_tol

# Check that the number of particles at the level weight is the same as predicted from analytic
# calculations
numparts_leveled = np.argmax(w > level_weight)  # This returns the first index for which
# w > level_weight, which thus corresponds to the number of particles at the level weight
expected_numparts_leveled = (
    numparts_init
    / (2.0 * t_r)
    * (
        1
        + erf(expected_mean_initial_weight * (t_r - 1.0))
        - 1.0
        / (np.sqrt(np.pi) * expected_mean_initial_weight)
        * np.exp(-((expected_mean_initial_weight * (t_r - 1.0)) ** 2))
    )
)
error = np.abs(numparts_leveled - expected_numparts_leveled)
std_numparts_leveled = np.sqrt(
    expected_numparts_leveled
    - numparts_init
    / np.sqrt(np.pi)
    / (t_r * expected_mean_initial_weight) ** 2
    * (
        np.sqrt(np.pi)
        / 4.0
        * (2.0 * expected_mean_initial_weight**2 + 1.0)
        * (1.0 - erf(expected_mean_initial_weight * (t_r - 1.0)))
        - 0.5
        * np.exp(
            -((expected_mean_initial_weight * (t_r - 1.0)) ** 2)
            * (expected_mean_initial_weight * (t_r + 1.0))
        )
    )
)
# 5 sigma test that has an intrinsic probability to fail of 1 over ~2 millions
print(
    "difference between expected and actual number of leveled particles (2nd species): "
    + str(error)
)
print("tolerance: " + str(5 * std_numparts_leveled))

numparts_unaffected = w.shape[0] - numparts_leveled
numparts_unaffected_anticipated = w0.shape[0] - np.argmax(w0 > level_weight)
# Check that number of particles with weight higher than level weight is the same before and after
# resampling
assert numparts_unaffected == numparts_unaffected_anticipated
# Check that particles with weight higher than level weight are unaffected by resampling.
assert np.all(w[-numparts_unaffected:] == w0[-numparts_unaffected:])
