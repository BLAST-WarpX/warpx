#!/usr/bin/env python
"""
This script tests the last coordinate after adding an electron.
The sphere is centered on O and has a radius of 0.2 (EB)
The electron is initially at: (0,0,-0.25) and moves with a velocity:
(0.5e10,0,1.0e10) with a time step of 1e-11.
An input file inputs_test_rz_particle_boundary_interaction_picmi.py is used.
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

# Analytical results
# ------------------
# Parameters read from warpx_used_inputs:
# the sphere (embedded boundary) is centered at the origin and has radius R;
# a single electron starts at (x0, 0, z0) with relativistic proper velocity
# (gamma*v) (ux0, 0, uz0). Since uy0 = 0 and y0 = 0, the motion is confined
# to the (x, z) plane.
# Note: WarpX stores proper velocities normalized by c in warpx_used_inputs.
input_dict = parse_input_file("./warpx_used_inputs")
R = float(input_dict["my_constants.radius"][0])
x0 = float(input_dict["electrons.multiple_particles_pos_x"][0])
z0 = float(input_dict["electrons.multiple_particles_pos_z"][0])
ux0 = float(input_dict["electrons.multiple_particles_ux"][0]) * c
uz0 = float(input_dict["electrons.multiple_particles_uz"][0]) * c

# The electron is relativistic, so we first compute the Lorentz factor and
# the corresponding (constant) 3-velocity from the proper velocity.
gamma = np.sqrt(1.0 + (ux0**2 + uz0**2) / c**2)
vx0, vz0 = ux0 / gamma, uz0 / gamma

# Step 1: find the time at which the electron hits the sphere.
# Before the impact, the trajectory is a straight line:
#     (x(t), z(t)) = (x0 + vx0*t, z0 + vz0*t).
# The impact condition x(t)^2 + z(t)^2 = R^2 yields the quadratic
#     (vx0^2 + vz0^2) * t^2 + 2*(x0*vx0 + z0*vz0) * t + (x0^2 + z0^2 - R^2) = 0.
# The first impact corresponds to the smaller root.
a_q = vx0**2 + vz0**2
b_q = 2.0 * (x0 * vx0 + z0 * vz0)
c_q = x0**2 + z0**2 - R**2
t_impact = (-b_q - np.sqrt(b_q**2 - 4 * a_q * c_q)) / (2 * a_q)

# Position of the point of impact on the sphere.
x_impact = x0 + vx0 * t_impact
z_impact = z0 + vz0 * t_impact

# Step 2: mirror-reflect the velocity at the impact point.
# The outward normal to the sphere at the impact point is r_impact / R.
# A mirror reflection gives v_reflected = v - 2 (v . n) n.
nx, nz = x_impact / R, z_impact / R
v_dot_n = vx0 * nx + vz0 * nz
vx_r = vx0 - 2 * v_dot_n * nx
vz_r = vz0 - 2 * v_dot_n * nz

# Step 3: propagate the reflected electron in a straight line until the end
# of the simulation. The remaining time is taken from the diagnostics so that
# the formula automatically adapts if the simulation duration changes.
t_remain = ts.t[-1] - t_impact
x_analytic = x_impact + vx_r * t_remain
y_analytic = 0.0
z_analytic = z_impact + vz_r * t_remain

print("NUMERICAL coordinates of the point of contact:")
print(f"x={x[0]:5.5f}, y={y[0]:5.5f}, z={z[0]:5.5f}")
print("\n")
print("ANALYTICAL coordinates of the point of contact:")
print(f"x={x_analytic:5.5f}, y={y_analytic:5.5f}, z={z_analytic:5.5f}")

tolerance = 0.02

rel_err_x = np.abs((x[0] - x_analytic) / x_analytic)
rel_err_z = np.abs((z[0] - z_analytic) / z_analytic)

print("\n")
print(f"Relative percentage error for x = {rel_err_x * 100:5.4f} %")
print(f"Relative percentage error for z = {rel_err_z * 100:5.4f} %")

assert (rel_err_x < tolerance) and (y[0] < 1e-8) and (rel_err_z < tolerance), (
    "Test particle_boundary_interaction did not pass"
)
