#!/usr/bin/env python3
#
# Copyright 2021-2022 Elisa Rheaume
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
This script tests the accuracy of the FieldProbe diagnostic by observing a plane
wave undergoing single slit diffraction. The input file inputs_2d is used. This
file defines the simulation box, laser pulse, embedded boundary with single slit,
and line of detector points. The plane wave initializes near the negative Z end
of the simulation box. The wave interacts with the embedded boundary at Z=0. The
wave undergoes diffraction at the slit. The electromagnetic flux is calculated
at the line detector which is placed perpendicular to Z beyond the slit. This
test will check if the detected EM flux matches expected values,
which can be solved analytically.
"""

import numpy as np
import pandas as pd

filename = "diags/reducedfiles/FP_line.txt"

# Open data file
df = pd.read_csv(filename, sep=" ")
df = df.sort_values(by=["[2]part_x_lev0-(m)"])

# Select position and Intensity of timestep 500
x = df.query("`[0]step()` == 500")["[2]part_x_lev0-(m)"]
S = df.query("`[0]step()` == 500")["[11]part_S_lev0-(W*s/m^2)"]
xvals = x.to_numpy()
svals = S.to_numpy()

# Default intensity is highest measured value for plane
# wave interacting with single slit
I_0 = np.max(S)


def I_envelope(x, lam=0.2e-6, a=0.3e-6, D=1.7e-6):
    arg = np.pi * a / lam * np.sin(np.arctan(x / D))
    return np.sinc(arg / np.pi) ** 2


# Count non-outlier values away from simulation boundaries
counter = np.arange(60, 140, 2)

# Count average error from expected values
error = 0
for a in counter:
    b = I_0 * I_envelope(xvals[a])
    c = svals[a]
    error += abs((c - b) / b) * 100.0
averror = error / (len(counter) - 1)

# average error range set at 2.5%
if averror > 2.5:
    print("Average error greater than 2.5%")

assert averror < 2.5
