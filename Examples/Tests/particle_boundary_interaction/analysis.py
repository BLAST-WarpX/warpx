#!/usr/bin/env python
# @Eya Dammak supervised by @Remi Lehe, 2024
"""
This script tests the last coordinate after adding an electron.
The sphere is centered on O and has a radius of 0.2 (EB)
The electron is initially at: (0,0,-0.25) and moves with a velocity:
(0.5e10,0,1.0e10) with a time step of 1e-11.
An input file inputs_test_rz_particle_boundary_interaction_picmi.py is used.
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

# Analytical results calculated
x_analytic = 0.03532
y_analytic = 0.00000
z_analytic = -0.20531

print("NUMERICAL coordinates of the point of contact:")
print(f"x={x[0]:5.5f}, y={y[0]:5.5f}, z={z[0]:5.5f}")
print("\n")
print("ANALYTICAL coordinates of the point of contact:")
print(f"x={x_analytic:5.5f}, y={y_analytic:5.5f}, z={z_analytic:5.5f}")

tolerance = 1e-5

rel_err_x = np.abs((x[0] - x_analytic) / x_analytic)
rel_err_z = np.abs((z[0] - z_analytic) / z_analytic)

print("\n")
print(f"Relative percentage error for x = {rel_err_x * 100:5.4f} %")
print(f"Relative percentage error for z = {rel_err_z * 100:5.4f} %")

assert (rel_err_x < tolerance) and (y[0] < 1e-8) and (rel_err_z < tolerance), (
    "Test particle_boundary_interaction did not pass"
)
