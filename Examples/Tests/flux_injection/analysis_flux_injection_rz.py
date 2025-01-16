#!/usr/bin/env python3
#
# Copyright 2021 Remi Lehe
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
This script tests the continuous injection of particles in
the azimuthal direction, in RZ geometry.

In this tests, relativistic electrons are continuously injected
in a uniform Bz field, at a radius that corresponds to their
Larmor radius. Therefore, throughout the simulations, these
electrons always stay at the same radius, while gyrating
around the z axis.

This script tests that:
- At the end of the simulation, the electrons are all at
  the correct radius. (This indirectly checks that the
  electrons were injected with the correct velocity
  throughout the simulation, and in particular that this
  velocity was along the azimuthal direction.)
- The total number of electrons corresponds to the expected flux.
"""

import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(0)

# Open plotfile specified in command line
fn = sys.argv[1]
ds = yt.load(fn)
t_max = ds.current_time.item()  # time of simulation

# Total number of electrons expected:
flux = 1.0  # in m^-2.s^-1, from the input script
emission_surface = 0.8  # in m^2,
# given that xmin = 1.5, xmax = 1.9, zmin = -1.0, zmax = 1.
n_tot = flux * emission_surface * t_max

# Read particle data
ad = ds.all_data()
r = ad["particle_position_x"].to_ndarray()  # Corresponds to the radial coordinate in RZ
w = ad["particle_weight"].to_ndarray()

# Check that the number of particles matches the expected one
assert np.allclose(w.sum(), n_tot, rtol=0.05)
# Check that the particles are at the right radius
assert np.all((r >= 1.48) & (r <= 1.92))
