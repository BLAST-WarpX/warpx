#!/usr/bin/env python
"""
This script checks that electron secondary emission (implemented by a callback function) works as intended.

In this test, four ions hit a spherical embedded boundary, and produce secondary
electrons with a probability of `0.4`. We thus expect ~2 electrons to be produced.
This script tests the number of electrons emitted and checks that their position is
close to the embedded boundary.
"""

import sys

sys.path.append("../../../Tools/Parser/")

import numpy as np
import yt
from input_file_parser import parse_input_file
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c

yt.funcs.mylog.setLevel(0)

# Open plotfile specified in command line
filename = sys.argv[1]
ts = OpenPMDTimeSeries(filename)

it = ts.iterations
x, y, z = ts.get_particle(["x", "y", "z"], species="electrons", iteration=it[-1])

N_sec_e = np.size(z)  # number of the secondary electrons

assert N_sec_e == 2, (
    "Test did not pass: for this set up we expect 2 secondary electrons emitted"
)

# Analytical results
# ------------------
# Parameters read from warpx_used_inputs:
# the sphere (embedded boundary) is centered at the origin and has radius R;
# four ions start at (xi, 0, zi) with proper velocity (gamma*v) (uxi, 0, uzi).
# Since uy0 = 0 and yi = 0, each ion travels in the (x, z) plane.
# When an ion reaches the sphere, it can emit a secondary electron at the
# impact point. The emitted electron then receives a small thermal kick
# (random direction, magnitude ~ sqrt(k Te / m_e)) and is propagated for
# the remaining fraction of the time step. Therefore the deterministic
# (analytical) part of the emitted electron position is the impact point
# of the ion on the sphere; the residual displacement comes from the
# thermal kick and from the embedded-boundary discretization.
# Note: WarpX stores proper velocities normalized by c in warpx_used_inputs.
input_dict = parse_input_file("./warpx_used_inputs")
R = float(input_dict["my_constants.radius"][0])
ion_x0 = np.array([float(v) for v in input_dict["ions.multiple_particles_pos_x"]])
ion_z0 = np.array([float(v) for v in input_dict["ions.multiple_particles_pos_z"]])
ion_ux0 = np.array([float(v) for v in input_dict["ions.multiple_particles_ux"]]) * c
ion_uz0 = np.array([float(v) for v in input_dict["ions.multiple_particles_uz"]]) * c

# The ions are non-relativistic (gamma ~ 1), but use the general formula
# to convert the proper velocity to the 3-velocity for consistency.
gamma = np.sqrt(1.0 + (ion_ux0**2 + ion_uz0**2) / c**2)
vx0 = ion_ux0 / gamma
vz0 = ion_uz0 / gamma

# For each ion, find the time at which it hits the sphere by solving
#     (vx0^2 + vz0^2) * t^2 + 2*(x0*vx0 + z0*vz0) * t + (x0^2 + z0^2 - R^2) = 0
# (the ray-sphere intersection in the (x, z) plane). The first impact
# corresponds to the smaller root.
a_q = vx0**2 + vz0**2
b_q = 2.0 * (ion_x0 * vx0 + ion_z0 * vz0)
c_q = ion_x0**2 + ion_z0**2 - R**2
t_impact = (-b_q - np.sqrt(b_q**2 - 4 * a_q * c_q)) / (2 * a_q)

# Coordinates of the four ion impact points on the sphere.
x_impact = ion_x0 + vx0 * t_impact
y_impact = np.zeros_like(x_impact)
z_impact = ion_z0 + vz0 * t_impact

# Each emitted electron is matched to the closest analytical impact point.
# The tolerance is set as an absolute distance (in meters) that bounds the
# combined effect of the random thermal kick and the EB discretization.
tolerance = 0.025

for i in range(0, N_sec_e):
    distances = np.sqrt(
        (x[i] - x_impact) ** 2 + (y[i] - y_impact) ** 2 + (z[i] - z_impact) ** 2
    )
    j = int(np.argmin(distances))
    print("\n")
    print(f"Electron # {i}:")
    print("NUMERICAL coordinates of the emitted electron:")
    print(f"x={x[i]:5.5f}, y={y[i]:5.5f}, z={z[i]:5.5f}")
    print("\n")
    print("ANALYTICAL coordinates of the closest ion impact point on the sphere:")
    print(
        f"(ion # {j}) x={x_impact[j]:5.5f}, y={y_impact[j]:5.5f}, z={z_impact[j]:5.5f}"
    )
    print(f"Distance to impact point = {distances[j]:.5f} m (tolerance: {tolerance} m)")

    assert distances[j] < tolerance, (
        f"Electron {i} is too far from any ion impact point on the sphere"
    )
