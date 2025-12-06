#!/usr/bin/env python3
#
# Copyright 2023 David Grote
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
This script tests the Gaussian-flux injection

The input files setup a uniform plasma with a drift and use flux injection
to attempt to maintain the constant density.
"""
import numpy as np
import openpmd_viewer
from scipy.constants import c, e

e_mass = 100.*m_e
i_mass = m_p
N = 1.e25
T = 1.

nz = 200
L = 2.e-7

def calcdensity(species, it):
    nn, info = ts.get_field(f"nn_{species}", iteration=it)
    return nn / dz, info


def calctemperature(species, mass, it):
    "Returns temperature in eV"
    nn, info = ts.get_field(f"nn_{species}", iteration=it)
    vx, info = ts.get_field(f"vx_{species}", iteration=it)
    vy, info = ts.get_field(f"vy_{species}", iteration=it)
    vz, info = ts.get_field(f"vz_{species}", iteration=it)
    vxvx, info = ts.get_field(f"vxvx_{species}", iteration=it)
    vyvy, info = ts.get_field(f"vyvy_{species}", iteration=it)
    vzvz, info = ts.get_field(f"vzvz_{species}", iteration=it)
    nn1 = nn.clip(1)
    T = (
        mass
        / 3.0
        * (vxvx - vx * vx / nn1 + vyvy - vy * vy / nn1 + vzvz - vz * vz / nn1)
        * c**2
        / nn1
    )
    return T / e, info

ts = openpmd_viewer.OpenPMDTimeSeries('diags/diag1')

it = 100

nn_electrons, info = calcdensity("electrons", it)
nn_ions, info = calcdensity("ions", it)

T_electrons, info = calctemperature("electrons", e_mass, it)
T_ions, info = calctemperature("ions", i_mass, it)

nn_electrons_error = np.abs((nn_electrons / N - 1.0))
nn_ions_error = np.abs((nn_ions / N - 1.0))
T_electrons_error = np.abs((T_electrons / T - 1.0))
T_ions_error = np.abs((T_ions / T - 1.0))

print(f"nn_electrons_error.max() = {nn_electrons_error.max()}")
print(f"nn_ions_error.max() = {nn_ions_error.max()}")
print(f"T_electrons_error.max() = {T_electrons_error.max()}")
print(f"T_ions_error.max() = {T_ions_error.max()}")

assert nn_electrons_error.max() < 0.1
assert nn_ions_error.max() < 0.1
assert T_electrons_error.max() < 0.1
assert T_ions_error.max() < 0.1
