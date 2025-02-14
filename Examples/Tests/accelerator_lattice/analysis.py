#!/usr/bin/env python3

# Copyright 2022 David Grote
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


"""
This script tests the quad lattice
The input file sets up a series of quadrupoles and propagates two particles through them.
One particle is in the X plane, the other the Y plane.
The final positions are compared to the analytic solutions.
The motion is slow enough that relativistic effects are ignored.
"""

import sys

import numpy as np
import yt
from scipy.constants import c, e, m_e

yt.funcs.mylog.setLevel(0)

filename = sys.argv[1]
ds = yt.load(filename)
ad = ds.all_data()

gamma_boost = float(ds.parameters.get("warpx.gamma_boost", 1.0))
uz_boost = np.sqrt(gamma_boost * gamma_boost - 1.0) * c

# Fetch the final particle position
xx_sim = ad["electron", "particle_position_x"].v[0]
zz_sim = ad["electron", "particle_position_z"].v[0]
ux_sim = ad["electron", "particle_momentum_x"].v[0] / m_e

if gamma_boost > 1.0:
    # The simulation data is in the boosted frame.
    # Transform the z position to the lab frame.
    time = ds.current_time.value
    zz_sim = gamma_boost * zz_sim + uz_boost * time

# Fetch the quadrupole lattice data
quad_starts = []
quad_lengths = []
quad_strengths_E = []
z_location = 0.0


def read_lattice(rootname, z_location):
    lattice_elements = ds.parameters.get(f"{rootname}.elements").split()
    for element in lattice_elements:
        element_type = ds.parameters.get(f"{element}.type")
        if element_type == "drift":
            length = float(ds.parameters.get(f"{element}.ds"))
            z_location += length
        elif element_type == "quad":
            length = float(ds.parameters.get(f"{element}.ds"))
            quad_starts.append(z_location)
            quad_lengths.append(length)
            quad_strengths_E.append(float(ds.parameters.get(f"{element}.dEdx")))
            z_location += length
        elif element_type == "line":
            z_location = read_lattice(element, z_location)
    return z_location


read_lattice("lattice", z_location)

# Fetch the initial position of the particle
x0 = [float(x) for x in ds.parameters.get("electron.single_particle_pos").split()]
ux0 = [float(x) * c for x in ds.parameters.get("electron.single_particle_u").split()]

xx = x0[0]
zz = x0[2]
ux = ux0[0]
uz = ux0[2]

gamma = np.sqrt(uz**2 / c**2 + 1.0)
vz = uz / gamma


def applylens(x0, vx0, vz0, gamma, lens_length, lens_strength):
    """Use analytic solution of a particle with a transverse dependent field"""
    kb0 = np.sqrt(e / (m_e * gamma * vz0**2) * abs(lens_strength))
    if lens_strength >= 0.0:
        x1 = x0 * np.cos(kb0 * lens_length) + (vx0 / vz0) / kb0 * np.sin(
            kb0 * lens_length
        )
        vx1 = vz0 * (
            -kb0 * x0 * np.sin(kb0 * lens_length)
            + (vx0 / vz0) * np.cos(kb0 * lens_length)
        )
    else:
        x1 = x0 * np.cosh(kb0 * lens_length) + (vx0 / vz0) / kb0 * np.sinh(
            kb0 * lens_length
        )
        vx1 = vz0 * (
            +kb0 * x0 * np.sinh(kb0 * lens_length)
            + (vx0 / vz0) * np.cosh(kb0 * lens_length)
        )
    return x1, vx1


# Integrate the particle using the analytic solution
for i in range(len(quad_starts)):
    z_lens = quad_starts[i]
    vx = ux / gamma
    dt = (z_lens - zz) / vz
    xx = xx + dt * vx
    xx, vx = applylens(xx, vx, vz, gamma, quad_lengths[i], quad_strengths_E[i])
    ux = gamma * vx
    zz = z_lens + quad_lengths[i]

dt = (zz_sim - zz) / vz
vx = ux / gamma
xx = xx + dt * vx

# Compare the analytic to the simulated final values
print(
    f"Error in x position is {abs(np.abs((xx - xx_sim) / xx))}, which should be < 0.01"
)
print(
    f"Error in x velocity is {abs(np.abs((ux - ux_sim) / ux))}, which should be < 0.002"
)

assert abs(np.abs((xx - xx_sim) / xx)) < 0.01, Exception("error in x particle position")
assert abs(np.abs((ux - ux_sim) / ux)) < 0.002, Exception(
    "error in x particle velocity"
)
