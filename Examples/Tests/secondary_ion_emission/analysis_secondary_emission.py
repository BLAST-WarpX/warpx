#!/usr/bin/env python
"""
This script checks the built-in secondary-emission model for absorbed particles.

In this test, four ions hit a spherical embedded boundary and produce secondary
electrons with a fixed secondary-emission probability. The emitted electrons are
expected to originate near the ion impact points on the embedded boundary.
"""

import sys

sys.path.append("../../../Tools/Parser/")

import numpy as np
from input_file_parser import parse_input_file
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c as clight

filename = sys.argv[1]
ts = OpenPMDTimeSeries(filename)

it = ts.iterations
x, y, z = ts.get_particle(["x", "y", "z"], species="electrons", iteration=it[-1])

N_sec_e = np.size(z)
expected_N_sec_e = 3
assert N_sec_e == expected_N_sec_e, (
    f"Expected {expected_N_sec_e} secondary electron, got {N_sec_e}"
)

input_dict = parse_input_file("./warpx_used_inputs")
R = float(input_dict["my_constants.radius"][0])
ion_x0 = np.array([eval(v) for v in input_dict["ions.multiple_particles_pos_x"]])
ion_z0 = np.array([eval(v) for v in input_dict["ions.multiple_particles_pos_z"]])
ion_ux0 = np.array([eval(v) for v in input_dict["ions.multiple_particles_ux"]]) * clight
ion_uz0 = np.array([eval(v) for v in input_dict["ions.multiple_particles_uz"]]) * clight

gamma = np.sqrt(1.0 + (ion_ux0**2 + ion_uz0**2) / clight**2)
vx0 = ion_ux0 / gamma
vz0 = ion_uz0 / gamma

a_q = vx0**2 + vz0**2
b_q = 2.0 * (ion_x0 * vx0 + ion_z0 * vz0)
c_q = ion_x0**2 + ion_z0**2 - R**2
t_impact = (-b_q - np.sqrt(b_q**2 - 4 * a_q * c_q)) / (2 * a_q)

x_impact = ion_x0 + vx0 * t_impact
y_impact = np.zeros_like(x_impact)
z_impact = ion_z0 + vz0 * t_impact

tolerance = 0.02

for i in range(0, N_sec_e):
    distances = np.array(
        [
            np.sqrt(
                (x[i] - x_impact[j]) ** 2
                + (y[i] - y_impact[j]) ** 2
                + (z[i] - z_impact[j]) ** 2
            )
            for j in range(len(x_impact))
        ]
    )
    j = int(np.argmin(distances))
    rel_dist = distances[j] / R
    print("\n")
    print(f"Electron # {i}:")
    print("NUMERICAL coordinates of the emitted electron:")
    print(f"x={x[i]:5.5f}, y={y[i]:5.5f}, z={z[i]:5.5f}")
    print("\n")
    print("ANALYTICAL coordinates of the closest ion impact point on the sphere:")
    print(
        f"(ion # {j}) x={x_impact[j]:5.5f}, y={y_impact[j]:5.5f}, z={z_impact[j]:5.5f}"
    )
    print(
        f"Relative distance to impact point = {rel_dist * 100:5.4f} % (tolerance: {tolerance * 100} %)"
    )

    assert rel_dist < tolerance, (
        f"Electron {i} is too far from any ion impact point on the sphere"
    )
