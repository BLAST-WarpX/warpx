#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Roelof Groenewald (Realta Fusion)
#
# License: BSD-3-Clause-LBNL

## In this test, a cold ring of particles rotating about the z-axis is resampled with
## velocity coincidence thinning in RZ geometry. In the local cylindrical frame all
## particles have the same velocity, so each cell must be merged into exactly two
## particles, and the merged particles must still move purely azimuthally.

import sys

import numpy as np
import yt
from scipy.constants import c, m_e

fn_final = sys.argv[1]
fn0 = fn_final[:-4] + "0000"

u0 = 0.1  # u_theta / c of all particles
num_filled_cells = 4

rel_tol = 1e-12
# the energy correction of the merge introduces velocity errors of order sqrt(machine eps)
vel_tol = 1e-5


def get_particle_data(fn):
    ad = yt.load(fn).all_data()
    r = ad["ring", "particle_position_x"].to_ndarray()
    theta = ad["ring", "particle_theta"].to_ndarray()
    w = ad["ring", "particle_weight"].to_ndarray()
    ux = ad["ring", "particle_momentum_x"].to_ndarray() / (m_e * c)
    uy = ad["ring", "particle_momentum_y"].to_ndarray() / (m_e * c)
    uz = ad["ring", "particle_momentum_z"].to_ndarray() / (m_e * c)
    ur = ux * np.cos(theta) + uy * np.sin(theta)
    ut = -ux * np.sin(theta) + uy * np.cos(theta)
    kinetic_energy = np.sum(w * (np.sqrt(1.0 + ux**2 + uy**2 + uz**2) - 1.0))
    angular_momentum = np.sum(w * r * ut)
    return w, ur, ut, uz, kinetic_energy, angular_momentum


w0, ur0, ut0, uz0, ke0, lz0 = get_particle_data(fn0)
w, ur, ut, uz, ke, lz = get_particle_data(fn_final)

print(f"number of particles: {w0.size} -> {w.size}")
print(f"max |u_r| / u0 = {np.max(np.abs(ur)) / u0:.3e}")
print(f"max |u_z| / u0 = {np.max(np.abs(uz)) / u0:.3e}")
print(f"max |u_theta / u0 - 1| = {np.max(np.abs(ut / u0 - 1.0)):.3e}")

# Each filled cell is a single velocity cluster that is merged into two particles
assert w.size == 2 * num_filled_cells

# Weight, kinetic energy and angular momentum about the z-axis are conserved
assert np.isclose(np.sum(w), np.sum(w0), rtol=rel_tol, atol=0.0)
assert np.isclose(ke, ke0, rtol=rel_tol, atol=0.0)
assert np.isclose(lz, lz0, rtol=rel_tol, atol=0.0)

# The merged particles still move purely azimuthally
assert np.max(np.abs(ur)) < vel_tol * u0
assert np.max(np.abs(uz)) < vel_tol * u0
assert np.max(np.abs(ut / u0 - 1.0)) < vel_tol
