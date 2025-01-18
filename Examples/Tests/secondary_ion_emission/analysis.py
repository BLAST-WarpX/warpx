#!/usr/bin/env python
"""
This script tests the last coordinates of the emitted secondary electrons.
The EB sphere is centered on O and has a radius of 0.2.
The proton is initially at: (0,0,-0.25) and moves with a velocity:
(0.1e6,0,1.5e6) with a time step of dt = 0.000000015.
The simulation uses a fixed random seed (np.random.seed(10025015))
to ensure the emission of secodnary electrons.
An input file inputs_test_rz_secondary_ion_emission_picmi.py is used.
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
print("x", x)
print("y", y)
print("z", z)
# Analytical results calculated
x_analytic = [0.004028, 0.003193]
y_analytic = [-0.0001518, -0.0011041]
z_analytic = [-0.19967, -0.19926]

N_sec_e = np.size(z)  # number of the secondary electrons

assert N_sec_e == 2, (
    "Test did not pass: for this set up we expect 2 secondary electrons emitted"
)

tolerance = 1e-3

for i in range(0, N_sec_e):
    print("\n")
    print(f"Electron # {i}:")
    print("NUMERICAL coordinates of the emitted electrons:")
    print("x=%5.5f, y=%5.5f, z=%5.5f" % (x[i], y[i], z[i]))
    print("\n")
    print("ANALYTICAL coordinates of the point of contact:")
    print("x=%5.5f, y=%5.5f, z=%5.5f" % (x_analytic[i], y_analytic[i], z_analytic[i]))

    diff_x = np.abs((x[i] - x_analytic[i]) / x_analytic[i])
    diff_y = np.abs((y[i] - y_analytic[i]) / y_analytic[i])
    diff_z = np.abs((z[i] - z_analytic[i]) / z_analytic[i])

    print("\n")
    print("percentage error for x = %5.4f %%" % (diff_x * 100))
    print("percentage error for y = %5.4f %%" % (diff_y * 100))
    print("percentage error for z = %5.4f %%" % (diff_z * 100))

    assert (diff_x < tolerance) and (diff_y < tolerance) and (diff_z < tolerance), (
        "Test particle_boundary_interaction did not pass"
    )
