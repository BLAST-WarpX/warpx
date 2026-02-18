#!/usr/bin/env python3
#
# Copyright 2021 Remi Lehe
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
This script tests the interaction between 2 electron macroparticles.

The macroparticles are moving towards each other and slow down due to the
repelling force between charges of the same sign. The particles should
be moving slow enough for relativistic effects to be negligible. In
this case the conservation of energy gives:

beta(t)^2 = beta(0)^2 + 2 * w * re * log(d(0)/d(t))

where:
re is the classical electron radius
w is the weight of the individual macroparticles
d is the distance between them
beta is the velocity normalized by the speed of light
"""

import glob
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import yt
from scipy.constants import c, m_e, physical_constants

yt.funcs.mylog.setLevel(0)

# Check plotfile name specified in command line
last_filename = sys.argv[1]
filename_radical = re.findall(r"(.*?)\d+/*$", last_filename)[0]

# Loop through files, and extract the position and velocity of both particles
x1 = []
x2 = []
beta1 = []
beta2 = []
for filename in sorted(glob.glob(filename_radical + "*")):
    print(filename)
    ds = yt.load(filename)
    ad = ds.all_data()

    x1.append(float(ad[("electron1", "particle_position_x")][0]))
    x2.append(float(ad[("electron2", "particle_position_x")][0]))
    beta1.append(float(ad[("electron1", "particle_momentum_x")][0]) / (m_e * c))
    beta2.append(float(ad[("electron2", "particle_momentum_x")][0]) / (m_e * c))

# Convert to numpy array
x1 = np.array(x1)
x2 = np.array(x2)
beta1 = np.array(beta1)
beta2 = np.array(beta2)

# Plot velocities, compare with theory
w = 5.0e12
re = physical_constants["classical electron radius"][0]
beta_th = np.sqrt(beta1[0] ** 2 - 2 * w * re * np.log((x2[0] - x1[0]) / (x2 - x1)))
plt.plot(beta1, "+", label="Particle 1")
plt.plot(-beta2, "x", label="Particle 2")
plt.plot(beta_th, "*", label="Theory")
plt.legend(loc=0)
plt.xlabel("Time (a.u.)")
plt.ylabel("Normalized velocity")
plt.savefig("Comparison.png")

# Check that the results are close to the theory
assert np.allclose(beta1[1:], beta_th[1:], atol=0.01)
assert np.allclose(-beta2[1:], beta_th[1:], atol=0.01)
