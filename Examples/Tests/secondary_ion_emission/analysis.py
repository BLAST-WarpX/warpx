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

N_sec_e = np.size(z)  # number of the secondary electrons

assert N_sec_e == 2, (
    "Test did not pass: for this set up we expect 2 secondary electrons emitted"
)

radius = 0.2
tolerance = 2e-2

# Check that the secondary electrons are on the surface of the sphere
for i in range(0, N_sec_e):
    print("\n")
    r = np.sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2)
    print(f"Electron # {i}: r={r:5.5f}")
    assert r - radius > 0, "Electron is not outside the sphere"
    assert r - radius < tolerance, "Electron is not on the surface of the sphere"
