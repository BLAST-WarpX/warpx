#!/usr/bin/env python3

# Copyright 2025 David Grote
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import sys

import numpy as np
import yt

# scipy.constants use CODATA2022
# from scipy.constants import e, m_e, c

# These are CODATA2018 values, as used in WarpX
e = 1.602176634e-19
m_e = 9.1093837015e-31
c = 299792458.0

# Integrate the test particle 5 timesteps, ending up with the position
# and velocity synchronized.
# In the simulation, with the synchronize_velocity_for_diagnostics flag set,
# the velocity will be synchronized at the end of step 5 when the diagnostics
# are written (even though that is not the final time step).

z = 0.1
uz = 0.0
Ez = -1.0
dt = 1.0e-6

# Half backward advance of velocity
uz -= -e / m_e * Ez * dt / 2.0

# Leap frog advance
for _ in range(5):
    uz += -e / m_e * Ez * dt
    g = np.sqrt((uz / c) ** 2 + 1.0)
    z += (uz / g) * dt

# Add half v advance to synchronize
uz += -e / m_e * Ez * dt / 2.0

filename = sys.argv[1]
ds = yt.load(filename)
ad = ds.all_data()
z_sim = ad["electron", "particle_position_x"]
uz_sim = ad["electron", "particle_momentum_z"] / m_e

print(f"Analysis   Z = {z:18.16f}, Uz = {uz:18.10f}")
print(f"Simulation Z = {z_sim.v[0]:18.16f}, Uz = {uz_sim.v[0]:18.10f}")

tolerance_rel = 1.0e-15
error_rel = np.abs((uz - uz_sim.v[0]) / uz)

print("error_rel    : " + str(error_rel))
print("tolerance_rel: " + str(tolerance_rel))

assert error_rel < tolerance_rel
