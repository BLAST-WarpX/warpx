#!/usr/bin/env python3
#
# A proton gyrates in a uniform B_z(t) = B0*(1 + t/t_ramp) applied to the
# particles. The theta-implicit push (theta = 1/2) rotates the momentum by
# 2*atan(omega*dt/2) per step, with omega evaluated at the stage time
# t^n + dt/2. The final momentum direction is compared with the sum of these
# rotations.
#
# Possible errors: 3e-15 rad
# tolerance: 1e-10 rad
# Evaluating the field at t^n instead gives an error of 4.8e-3 rad, and
# missing the field entirely gives an error of 2.9 rad.

import sys

import numpy as np
import yt
from scipy.constants import c, e, m_p

max_steps = 200
dt = 1.0e-8
B0 = 0.01
t_ramp = max_steps * dt
u0 = 1.0e4
theta = 0.5

ds = yt.load(sys.argv[1])
ad = ds.all_data()
px = ad["proton", "particle_momentum_x"].to_ndarray()
py = ad["proton", "particle_momentum_y"].to_ndarray()
assert px.size == 1, f"expected 1 proton, found {px.size}"

gamma = np.sqrt(1.0 + (u0 / c) ** 2)
t_stage = (np.arange(max_steps) + theta) * dt
omega = e * B0 * (1.0 + t_stage / t_ramp) / (m_p * gamma)
expected_angle = -np.sum(2.0 * np.arctan(omega * dt / 2.0))

angle = np.arctan2(py[0], px[0])
error = abs(np.angle(np.exp(1j * (angle - expected_angle))))
p_rel_error = abs(np.hypot(px[0], py[0]) / (m_p * u0) - 1.0)

print("angle error = ", error)
print("momentum magnitude error = ", p_rel_error)
assert error < 1.0e-10
assert p_rel_error < 1.0e-12
