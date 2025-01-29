#!/usr/bin/env python3

# Copyright 2022 David Grote
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This script tests the ion stopping model.
# It uses the same stopping power formula that
# is used in the C++ to check the resulting
# particle energies.

import sys

import numpy as np
import yt
from scipy.constants import e, epsilon_0, k, m_e, m_p

# Define constants using the WarpX names for the evals below
q_e = e
kb = k

# Tolerance on the error in the final energy (in eV)
tolerance = 1.0e-7

last_filename = sys.argv[1]
last_it = 10


def stopping_from_electrons(ne, Te, Zb, ion_mass):
    """Calculate the coefficient in equation 14.13 from
    "Introduction to Plasma Physics", Goldston and Rutherford.
    ne: electron density
    Te: electron temperature (eV)
    Zb: ion charge state
    ion_mass: (kg)
    """
    vthe = np.sqrt(3.0 * Te * e / m_e)
    wpe = np.sqrt(ne * e**2 / (epsilon_0 * m_e))
    lambdadb = vthe / wpe
    loglambda = np.log((12.0 * np.pi / Zb) * (ne * lambdadb**3))
    # Goldston's equation 14.13
    dEdt = (
        -np.sqrt(2.0)
        * ne
        * Zb**2
        * e**4
        * np.sqrt(m_e)
        * loglambda
        / (6.0 * np.pi**1.5 * epsilon_0**2 * ion_mass * (Te * e) ** 1.5)
    )
    return dEdt


def stopping_from_ions(dt, ni, Ti, mi, Zi, Zb, ion_mass, ion_energy):
    """
    ni: background ion density
    Ti: background ion temperature (eV)
    mi: background ion mass
    Zi: background ion charge state
    Zb: ion charge state
    ion_mass: (kg)
    ion_energy: (eV)
    """
    vthi = np.sqrt(3.0 * Ti * e / mi)
    wpi = np.sqrt(ni * e**2 / (epsilon_0 * mi))
    lambdadb = vthi / wpi
    loglambda = np.log((12.0 * np.pi / Zb) * (ni * lambdadb**3))
    alpha = (
        np.sqrt(2.0)
        * ni
        * Zi**2
        * Zb**2
        * e**4
        * np.sqrt(ion_mass)
        * loglambda
        / (8.0 * np.pi * epsilon_0**2 * mi)
    )
    f1 = np.clip((ion_energy * e) ** (3.0 / 2.0) - 3.0 / 2.0 * alpha * dt, 0.0, None)
    ion_energy = (f1) ** (2.0 / 3.0) / e
    return ion_energy


# Fetch background parameters and initial particle data
ds0 = yt.load("diags/diag1000000")
ad0 = ds0.all_data()

Zb = 1.0  # Ion charge state

ne = float(ds0.parameters["stopping_on_electrons_constant.background_density"])
Te = (
    eval(ds0.parameters["stopping_on_electrons_constant.background_temperature"])
    * kb
    / e
)
ion_mass12 = m_p

mi = eval(ds0.parameters["stopping_on_ions_constant.background_mass"])
Zi = float(ds0.parameters["stopping_on_ions_constant.background_charge_state"])
ni = float(ds0.parameters["stopping_on_ions_constant.background_density"])
Ti = eval(ds0.parameters["stopping_on_ions_constant.background_temperature"]) * kb / e
ion_mass34 = 4.0 * m_p

# For ions1, the background parameters are constants
vx = ad0[("ions1", "particle_momentum_x")].to_ndarray() / ion_mass12
vy = ad0[("ions1", "particle_momentum_y")].to_ndarray() / ion_mass12
vz = ad0[("ions1", "particle_momentum_z")].to_ndarray() / ion_mass12
EE1 = 0.5 * ion_mass12 * (vx**2 + vy**2 + vz**2) / e

# For ions2, the background parameters are parsed
xx = ad0[("ions2", "particle_position_x")].to_ndarray() / ion_mass12
yy = ad0[("ions2", "particle_position_y")].to_ndarray() / ion_mass12
ne2 = np.where(xx > 0.0, 1.0e20, 1.0e21)
Te2 = np.where(yy > 0.0, 1.0, 2.0)

vx = ad0[("ions2", "particle_momentum_x")].to_ndarray() / ion_mass12
vy = ad0[("ions2", "particle_momentum_y")].to_ndarray() / ion_mass12
vz = ad0[("ions2", "particle_momentum_z")].to_ndarray() / ion_mass12
EE2 = 0.5 * ion_mass12 * (vx**2 + vy**2 + vz**2) / e

# For ions3, the background parameters are constants
vx = ad0[("ions3", "particle_momentum_x")].to_ndarray() / ion_mass34
vy = ad0[("ions3", "particle_momentum_y")].to_ndarray() / ion_mass34
vz = ad0[("ions3", "particle_momentum_z")].to_ndarray() / ion_mass34
EE3 = 0.5 * ion_mass34 * (vx**2 + vy**2 + vz**2) / e

# For ions4, the background parameters are parsed
xx = ad0[("ions4", "particle_position_x")].to_ndarray() / ion_mass34
yy = ad0[("ions4", "particle_position_y")].to_ndarray() / ion_mass34
ni4 = np.where(xx > 0.0, 1.0e20, 1.0e21)
Ti4 = np.where(yy > 0.0, 0.05, 0.10)

vx = ad0[("ions4", "particle_momentum_x")].to_ndarray() / ion_mass34
vy = ad0[("ions4", "particle_momentum_y")].to_ndarray() / ion_mass34
vz = ad0[("ions4", "particle_momentum_z")].to_ndarray() / ion_mass34
EE4 = 0.5 * ion_mass34 * (vx**2 + vy**2 + vz**2) / e


ds = yt.load(last_filename)
ad = ds.all_data()
dt = ds.current_time.to_value() / last_it

# Step through the same number of time steps
a_EE1 = EE1
a_EE2 = EE2
a_EE3 = EE3
a_EE4 = EE4
for it in range(last_it):
    dEdt1 = stopping_from_electrons(ne, Te, Zb, ion_mass12)
    a_EE1 *= np.exp(dEdt1 * dt)
    dEdt2 = stopping_from_electrons(ne2, Te2, Zb, ion_mass12)
    a_EE2 *= np.exp(dEdt2 * dt)
    a_EE3 = stopping_from_ions(dt, ni, Ti, mi, Zi, Zb, ion_mass34, a_EE3)
    a_EE4 = stopping_from_ions(dt, ni4, Ti4, mi, Zi, Zb, ion_mass34, a_EE4)

# Fetch the final particle data
vx = ad[("ions1", "particle_momentum_x")].to_ndarray() / ion_mass12
vy = ad[("ions1", "particle_momentum_y")].to_ndarray() / ion_mass12
vz = ad[("ions1", "particle_momentum_z")].to_ndarray() / ion_mass12
EE1 = 0.5 * ion_mass12 * (vx**2 + vy**2 + vz**2) / e

vx = ad[("ions2", "particle_momentum_x")].to_ndarray() / ion_mass12
vy = ad[("ions2", "particle_momentum_y")].to_ndarray() / ion_mass12
vz = ad[("ions2", "particle_momentum_z")].to_ndarray() / ion_mass12
EE2 = 0.5 * ion_mass12 * (vx**2 + vy**2 + vz**2) / e

vx = ad[("ions3", "particle_momentum_x")].to_ndarray() / ion_mass34
vy = ad[("ions3", "particle_momentum_y")].to_ndarray() / ion_mass34
vz = ad[("ions3", "particle_momentum_z")].to_ndarray() / ion_mass34
EE3 = 0.5 * ion_mass34 * (vx**2 + vy**2 + vz**2) / e

vx = ad[("ions4", "particle_momentum_x")].to_ndarray() / ion_mass34
vy = ad[("ions4", "particle_momentum_y")].to_ndarray() / ion_mass34
vz = ad[("ions4", "particle_momentum_z")].to_ndarray() / ion_mass34
EE4 = 0.5 * ion_mass34 * (vx**2 + vy**2 + vz**2) / e

error1 = np.abs(EE1 - a_EE1)
error2 = np.abs(EE2 - a_EE2)
error3 = np.abs(EE3 - a_EE3)
error4 = np.abs(EE4 - a_EE4)
print("stopping on electrons error with constant = ", error1)
print("stopping on electrons error with parsed = ", error2)
print("stopping on ions error with constant = ", error3)
print("stopping on ions error with parsed = ", error4)
print("tolerance = ", tolerance)

assert np.all(error1 < tolerance)
assert np.all(error2 < tolerance)
assert np.all(error3 < tolerance)
assert np.all(error4 < tolerance)
