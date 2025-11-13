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

import os
import sys

import numpy as np
import openpmd_viewer
from scipy.constants import c, e

sys.path.insert(1, "../../../../warpx/Regression/Checksum/")
import checksumAPI

sys.path.append("../../../../warpx/Tools/Parser/")
from input_file_parser import parse_input_file

input_dict = parse_input_file("inputs_1d_fixed")

e_mass = eval(input_dict["my_constants.e_mass"][0])
i_mass = eval(input_dict["my_constants.i_mass"][0])
N = float(input_dict["my_constants.N"][0])
T = eval(input_dict["my_constants.T"][0])

nz = eval(input_dict["my_constants.nz"][0])
L = eval(input_dict["my_constants.L"][0])

dz = L / nz


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


ts = openpmd_viewer.OpenPMDTimeSeries("FluxInjection1D_plt")

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

assert nn_electrons_error.max() < 0.03
assert nn_ions_error.max() < 0.03
assert T_electrons_error.max() < 0.03
assert T_ions_error.max() < 0.03

# Verify checksum
fn = sys.argv[1]
test_name = os.path.split(os.getcwd())[1]
checksumAPI.evaluate_checksum(test_name, fn, output_format="openpmd")
