#!/usr/bin/env python

"""
This script verifies that Gauss’s law is preserved during particle splitting.

"""

import sys

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import epsilon_0

# Analyze the diagnostics
filename = sys.argv[1]
ts = OpenPMDTimeSeries(filename)

print("Check Gauss's law at every iteration:")

tolerance = 1e-12

for iteration in ts.iterations[1:]:
    rho, _ = ts.get_field("rho", iteration=iteration)
    divE, _ = ts.get_field("divE", iteration=iteration)

    rho_over_eps0 = rho / epsilon_0

    error_rel = np.amax(np.abs(divE - rho_over_eps0)) / np.amax(np.abs(rho_over_eps0))

    print(
        f"iteration = {iteration}, "
        f"error_rel = {error_rel:.6e}, "
        f"tolerance = {tolerance:.6e}"
    )

    assert error_rel < tolerance, (
        f"Gauss's law violated at iteration {iteration}: "
        f"error_rel = {error_rel:.6e} >= {tolerance:.6e}"
    )
