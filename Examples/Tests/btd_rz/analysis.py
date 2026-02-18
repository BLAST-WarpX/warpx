#!/usr/bin/env python3

# Copyright 2022
# Authors: Revathi Jambunathan, Remi Lehe
#
# This tests checks the backtransformed diagnostics by emitting a laser
# (with the antenna) in the boosted-frame and then checking that the
# fields recorded by the backtransformed diagnostics have the right amplitude,
# wavelength, and envelope (i.e. gaussian envelope with the right duration.

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, e, m_e
from scipy.optimize import curve_fit


def gaussian_laser(z, a0, z0_phase, z0_prop, ctau, lambda0):
    """
    Returns a Gaussian laser profile
    """
    k0 = 2 * np.pi / lambda0
    E0 = a0 * m_e * c**2 * k0 / e
    return E0 * np.exp(-((z - z0_prop) ** 2) / ctau**2) * np.cos(k0 * (z - z0_phase))


# Fit the on-axis profile to extract the phase (a.k.a. CEP)
def fit_function(z, z0_phase):
    return gaussian_laser(z, a0, z0_phase, z0_b + Lprop_b, ctau0, lambda0)


# The values must be consistent with the values provided in the simulation input
t_current = 80e-15  # Time of the snapshot1
z0_antenna = -1.0e-6  # position of laser
lambda0 = 0.8e-6  # wavelength of the signal
tau0 = 10e-15  # duration of the signal
ctau0 = tau0 * c
a0 = 15  # amplitude
t_peak = 20e-15  # Time at which laser reaches its peak
Lprop_b = c * t_current
z0_b = z0_antenna - c * t_peak

ts = OpenPMDTimeSeries("./diags/back_rz")
Ex, info = ts.get_field("E", "x", iteration=1, slice_across="r")

fit_result = curve_fit(fit_function, info.z, Ex, p0=np.array([z0_b + Lprop_b]))
z0_fit = fit_result[0]

Ex_fit = gaussian_laser(info.z, a0, z0_fit, z0_b + Lprop_b, ctau0, lambda0)

## Check that the a0 agrees within 5% of the predicted value
assert np.allclose(Ex, Ex_fit, atol=0.18 * Ex.max())
