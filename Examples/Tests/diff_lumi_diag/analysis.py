#!/usr/bin/env python3
# This test checks the differential luminosity of the beam-beam interaction
# in the case of two Gaussian beams crossing rigidly at the interaction point.
# The beams have a Gaussian distribution both in energy and in transverse positions.
# In that case, the differential luminosity can be calculated analytically.

import os
import re

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

# Extract the 1D differential luminosity from the file
filename = "./diags/reducedfiles/DifferentialLuminosity_beam1_beam2.txt"
with open(filename) as f:
    # First line: header, contains the energies
    line = f.readline()
    E_bin = np.array(list(map(float, re.findall("=(.*?)\(", line))))
data = np.loadtxt(filename)
dE_bin = E_bin[1] - E_bin[0]
dL_dE_sim = data[-1, 2:]  # Differential luminosity at the end of the simulation

# Beam parameters
N = 1.2e10
E_beam = 125e9  # in eV
sigma_x = 500e-9
sigma_y = 10e-9

# Compute the analytical differential luminosity for 2 Gaussian beams
sigma_E1 = 0.02 * E_beam
sigma_E2 = 0.03 * E_beam
sigma_E = np.sqrt(
    sigma_E1**2 + sigma_E2**2
)  # Expected width of the differential luminosity
dL_dE_th = (
    N**2
    / (2 * (2 * np.pi) ** 1.5 * sigma_x * sigma_y * sigma_E)
    * np.exp(-((E_bin - 2 * E_beam) ** 2) / (2 * sigma_E**2))
)

# Extract the 2D differential luminosity from the file
series = OpenPMDTimeSeries("./diags/reducedfiles/DifferentialLuminosity2d_beam1_beam2/")
d2L_dE1_dE2_sim, info = series.get_field("d2L_dE1_dE2", iteration=80)

# Compute the analytical 2D differential luminosity for 2 Gaussian beams
assert info.axes[0] == "E2"
assert info.axes[1] == "E1"
E2, E1 = np.meshgrid(info.E2, info.E1, indexing="ij")
d2L_dE1_dE2_th = (
    N**2
    / (2 * (2 * np.pi) ** 2 * sigma_x * sigma_y * sigma_E1 * sigma_E2)
    * np.exp(
        -((E1 - E_beam) ** 2) / (2 * sigma_E1**2)
        - (E2 - E_beam) ** 2 / (2 * sigma_E2**2)
    )
)

# Extract test name from path
test_name = os.path.split(os.getcwd())[1]
print("test_name", test_name)

# Pick tolerance
if "leptons" in test_name:
    tol1 = 0.02
    tol2 = 0.04
elif "photons" in test_name:
    # In the photons case, the particles are
    # initialized from a density distribution ;
    # tolerance is larger due to lower particle statistics
    tol1 = 0.021
    tol2 = 0.06

# Check that the 1D diagnostic and analytical result match
error1 = abs(dL_dE_sim - dL_dE_th).max() / abs(dL_dE_th).max()
print("Relative error: ", error1)
print("Tolerance: ", tol1)

# Check that the 2D and 1D diagnostics match
error2 = abs(d2L_dE1_dE2_sim - d2L_dE1_dE2_th).max() / abs(d2L_dE1_dE2_th).max()
print("Relative error: ", error2)
print("Tolerance: ", tol2)

assert error1 < tol1
assert error2 < tol2
