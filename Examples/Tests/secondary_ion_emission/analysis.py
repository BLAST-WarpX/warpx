#!/usr/bin/env python
"""
This script checks that electron secondary emission (implemented by a callback function) works as intended.

In this test, four ions hit a spherical embedded boundary, and produce secondary
electrons with a probability of `0.4`. We thus expect ~2 electrons to be produced.
This script tests the number of electrons emitted and checks that their position is
close to the embedded boundary.
"""

import sys

import numpy as np
import yt
from openpmd_viewer import OpenPMDTimeSeries

yt.funcs.mylog.setLevel(0)

# Open plotfile specified in command line
filename = sys.argv[1]
ts = OpenPMDTimeSeries(filename)

it = ts.iterations
x, y, z = ts.get_particle(["x", "y", "z"], species="electrons", iteration=it[-1])

x_analytic = [-0.091696, 0.011599]
y_analytic = [-0.002282, -0.0111624]
z_analytic = [-0.200242, -0.201728]

N_sec_e = np.size(z)  # number of the secondary electrons

assert N_sec_e == 2, (
    "Test did not pass: for this set up we expect 2 secondary electrons emitted"
)

tolerance = 1e-3

for i in range(0, N_sec_e):
    print("\n")
    print(f"Electron # {i}:")
    print("NUMERICAL coordinates of the emitted electrons:")
    print(f"x={x[i]:5.5f}, y={y[i]:5.5f}, z={z[i]:5.5f}")
    print("\n")
    print("ANALYTICAL coordinates of the point of contact:")
    print(f"x={x_analytic[i]:5.5f}, y={y_analytic[i]:5.5f}, z={z_analytic[i]:5.5f}")

    rel_err_x = np.abs((x[i] - x_analytic[i]) / x_analytic[i])
    rel_err_y = np.abs((y[i] - y_analytic[i]) / y_analytic[i])
    rel_err_z = np.abs((z[i] - z_analytic[i]) / z_analytic[i])

    print("\n")
    print(f"Relative percentage error for x = {rel_err_x * 100:5.4f} %")
    print(f"Relative percentage error for y = {rel_err_y * 100:5.4f} %")
    print(f"Relative percentage error for z = {rel_err_z * 100:5.4f} %")

    assert (
        (rel_err_x < tolerance) and (rel_err_y < tolerance) and (rel_err_z < tolerance)
    ), "Test particle_boundary_interaction did not pass"
