#!/usr/bin/env python3

# Copyright 2021 David Grote
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


"""
This script tests the particle boundary conditions.
The input file sets up absorbing, periodic, and reflecting boundary conditions
along each of the three axis. It launches particles heading toward each of the boundaries
and checks that they end up in the correct place (or are deleted).
"""

import sys

import numpy as np
import yt
from scipy.constants import c, m_e

yt.funcs.mylog.setLevel(0)

# The min and max size of the box along the three axis.
dmin = -1.0
dmax = +1.0

# Open plotfile specified in command line
filename = sys.argv[1]
ds = yt.load(filename)
ad = ds.all_data()
time = ds.current_time.to_value()

filename0 = filename[:-5] + "00000"
ds0 = yt.load(filename0)
ad0 = ds0.all_data()

# Read in the particle initial values and the current values.
# They need to be sorted by their ids since they may be ordered
# differently in the diagnostic files.
# For the absorbing particles, an extra particle was added that won't be absorbed
# so that there will be something to read in here.
r_id0 = ad0["reflecting_particles", "particle_id"].v
a_id0 = ad0["absorbing_particles", "particle_id"].v
p_id0 = ad0["periodic_particles", "particle_id"].v

xx0 = ad0["reflecting_particles", "particle_position_x"].v[np.argsort(r_id0)]
zz0 = ad0["periodic_particles", "particle_position_z"].v[np.argsort(p_id0)]

ux0 = ad0["reflecting_particles", "particle_momentum_x"].v[np.argsort(r_id0)] / m_e / c
uz0 = ad0["periodic_particles", "particle_momentum_z"].v[np.argsort(p_id0)] / m_e / c
gx0 = np.sqrt(1.0 + ux0**2)
gz0 = np.sqrt(1.0 + uz0**2)
vx0 = ux0 / gx0 * c
vz0 = uz0 / gz0 * c

r_id = ad["reflecting_particles", "particle_id"].v
a_id = ad["absorbing_particles", "particle_id"].v
p_id = ad["periodic_particles", "particle_id"].v

xx = ad["reflecting_particles", "particle_position_x"].v[np.argsort(r_id)]
zz = ad["periodic_particles", "particle_position_z"].v[np.argsort(p_id)]

ux = ad["reflecting_particles", "particle_momentum_x"].v[np.argsort(r_id)] / m_e / c
uz = ad["periodic_particles", "particle_momentum_z"].v[np.argsort(p_id)] / m_e / c
gx = np.sqrt(1.0 + ux**2)
gz = np.sqrt(1.0 + uz**2)
vx = ux / gx * c
vz = uz / gz * c


def do_reflect(x):
    if x < dmin:
        return 2.0 * dmin - x
    elif x > dmax:
        return 2.0 * dmax - x
    else:
        return x


def do_periodic(x):
    if x < dmin:
        return x + (dmax - dmin)
    elif x > dmax:
        return x - (dmax - dmin)
    else:
        return x


# Calculate the analytic value of the current particle locations and
# apply the appropriate boundary conditions.
xxa = xx0 + vx0 * time
xxa[0] = do_reflect(xxa[0])
xxa[1] = do_reflect(xxa[1])

zza = zz0 + vz0 * time
zza[0] = do_periodic(zza[0])
zza[1] = do_periodic(zza[1])

assert len(a_id) == 1, "Absorbing particles not absorbed"
assert np.all(vx == -vx0), "Reflecting particle velocity not correct"
assert np.all(vz == +vz0), "Periodic particle velocity not correct"
assert np.all(
    np.abs((xx - xxa) / xx) < 1.0e-15
), "Reflecting particle position not correct"
assert np.all(
    np.abs((zz - zza) / zz) < 1.0e-15
), "Periodic particle position not correct"
