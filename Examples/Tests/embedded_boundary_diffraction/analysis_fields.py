#!/usr/bin/env python3
"""
This test checks the implementation of the embedded boundary in cylindrical geometry,
by checking the diffraction of a laser by an embedded boundary here.
We then check that the first minimum of the diffracted intensity pattern
occurs along the angle given by the theoretical Airy pattern, i.e.
theta_diffraction = 1.22 * lambda / d
"""

import sys

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.ndimage import gaussian_filter1d

filename = sys.argv[1]
ts = OpenPMDTimeSeries(filename)

# Extract the intensity as a function of r and z
Ex, info = ts.get_field("E", "x", iteration=300)
In = gaussian_filter1d(
    Ex**2, sigma=5, axis=0
)  # Extract intensity by averaging E^2 over wavelength
irmax = np.argmax(In, axis=-1)


# Find the radius of the first minimum, as a function of z
def r_first_minimum(iz):
    ir = len(info.r) // 2
    while In[iz, ir + 1] < In[iz, ir]:
        ir += 1
    return info.r[ir]


r = np.array([r_first_minimum(iz) for iz in range(len(info.z))])

# Check that this corresponds to the prediction from the Airy pattern
theta_diffraction = np.arcsin(1.22 * 0.1 / 0.4) / 2
assert np.all(abs(r[50:] - theta_diffraction * info.z[50:]) < 0.03)
