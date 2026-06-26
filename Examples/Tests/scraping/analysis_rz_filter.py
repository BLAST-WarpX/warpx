#!/usr/bin/env python

# Copyright 2023 Kale Weichman
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This script tests the particle scraping for the embedded boundary in RZ.
# Particles are initialized between r=0.15 and r=0.2
# having a negative radial velocity.
# A cylindrical embedded surface is placed at r=0.1.
# Upon reaching the surface, particles should be removed.
# At the end of the simulation, i.e., at time step 37,
# there should be 512 particles left.
# This test checks that plot_filter_fucntion works with the boundary scraping diagnostic
# by making sure that the particles removed from only half the domain (z>0) have been recorded.

# Possible errors: 0
# tolerance: 0
# Possible running time: < 1 s

import sys

import numpy as np
import yt
from openpmd_viewer import OpenPMDTimeSeries

tolerance = 0

fn = sys.argv[1]
ds = yt.load(fn)
ad = ds.all_data()
x = ad["electron", "particle_position_x"].v

error = len(x) - 512
print("error = ", error)
print("tolerance = ", tolerance)
assert error == tolerance

# Check that all the removed particles are properly recorded
# by making sure that, at each iteration, the sum of the number of
# remaining particles and scraped particles is equal to half the
# original number of particles
# also check that no particles with z <= 0 have been scraped
ts_full = OpenPMDTimeSeries("./diags/diag2/")
ts_scraping = OpenPMDTimeSeries("./diags/diag3/particles_at_eb")


def n_remaining_particles(iteration):
    (w,) = ts_full.get_particle(["w"], iteration=iteration)
    return len(w)


def n_scraped_particles(iteration):
    step_scraped = ts_scraping.get_particle(
        ["stepScraped"], iteration=ts_scraping.iterations[0]
    )
    return (step_scraped <= iteration).sum()


def n_scraped_z_leq_zero(iteration):
    (z_pos,) = ts_scraping.get_particle(["z"], iteration=ts_scraping.iterations[0])
    return (z_pos <= 0).sum()


n_remaining = np.array(
    [n_remaining_particles(iteration) for iteration in ts_full.iterations]
)
n_scraped = np.array(
    [n_scraped_particles(iteration) for iteration in ts_full.iterations]
)
n_z_leq_zero = np.array(
    [n_scraped_z_leq_zero(iteration) for iteration in ts_full.iterations]
)
n_total = n_remaining[0]
assert np.all(2 * n_scraped + n_remaining == n_total)
assert np.all(n_z_leq_zero == 0)
