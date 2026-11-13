#!/usr/bin/env python3

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c as clight
from scipy.constants import electron_mass as m_e
from scipy.constants import elementary_charge as q_e

# Expected temperature profile
r_from_file = np.linspace(0.0, 1.0, 16)
u_std_0 = 0.2 * (0.5 + r_from_file)
Te_from_file = u_std_0**2 * m_e * clight**2 / q_e

# Load particles
ts = OpenPMDTimeSeries("./diags/diag1")

x, y, z, ux, uy, uz = ts.get_particle(
    ["x", "y", "z", "ux", "uy", "uz"],
    species="electrons",
    iteration=0,
)


def check_standard_normal(u, mean_ref, std_ref, tolerance):
    r = (u - mean_ref) / std_ref
    r_mean = np.mean(r)
    r_std = np.std(r)
    print(" abs(r_mean) = ", abs(r_mean))
    assert abs(r_mean) < tolerance
    print(" abs(r_std - 1.0) = ", abs(r_std - 1.0))
    assert abs(r_std - 1.0) < tolerance


# Cartesian coordinates -> spherical radius
r_wx = np.sqrt(x**2 + y**2 + z**2)

# Interpolate temperature to particle positions
T_interp = np.interp(r_wx, r_from_file, Te_from_file)

# Expected normalized momentum standard deviation
u_std = np.sqrt(T_interp * q_e / (m_e * clight**2))

# Zero bulk velocity
u_mean = np.zeros_like(u_std)

tolerance = 0.05

# Check all three momentum components
for u in (ux, uy, uz):
    check_standard_normal(u, u_mean, u_std, tolerance)
