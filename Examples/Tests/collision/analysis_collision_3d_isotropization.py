#!/usr/bin/env python3

# This script tests the collision module
# using isotropization relaxation in 3D.
# Initially, electrons have different temperatures in different directions.
# Relaxation occurs to bring the two temperatures to be
# a final same temperature through collisions.
# The result is compared with an analytical solution.
# A good reference is given by the code Smilei:
# https://github.com/SmileiPIC/Smilei/tree/master/benchmarks/collisions
# https://smileipic.github.io/tutorials/advanced_collisions.html
# https://smileipic.github.io/Smilei/Understand/collisions.html#test-cases-for-collisions

import sys

import numpy as np
import scipy.constants as sc
import yt

e = sc.e
pi = sc.pi
ep0 = sc.epsilon_0
m = sc.m_e

dt = 1.4e-17
ne = 1.116e28
log = 2.0
T_par = 5.62 * e
T_per = 5.1 * e

A = 1.0 - T_per / T_par
mu = (
    e**4
    * ne
    * log
    / (8.0 * pi**1.5 * ep0**2 * m**0.5 * T_par**1.5)
    * A ** (-2)
    * (-3.0 + (3.0 - A) * np.arctanh(A**0.5) / A**0.5)
)

fn = sys.argv[1]
ds = yt.load(fn)
ad = ds.all_data()
vx = ad["electron", "particle_momentum_x"].to_ndarray() / m
vy = ad["electron", "particle_momentum_y"].to_ndarray() / m
Tx = np.mean(vx**2) * m / e
Ty = np.mean(vy**2) * m / e

nt = 100
Tx0 = T_par
Ty0 = T_per
for _ in range(nt - 1):
    Tx0 = Tx0 + dt * mu * (Ty0 - Tx0) * 2.0
    Ty0 = Ty0 + dt * mu * (Tx0 - Ty0)

tolerance = 0.05
error = np.maximum(abs(Tx - Tx0 / e) / Tx, abs(Ty - Ty0 / e) / Ty)

print(f"error = {error}")
print(f"tolerance = {tolerance}")
assert error < tolerance
