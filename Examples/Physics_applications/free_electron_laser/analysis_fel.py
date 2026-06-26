#!/usr/bin/env python

"""
This script tests that the FEL is correctly modelled in the simulation.

The physical parameters are the same as the ones from section 5.1
of https://arxiv.org/pdf/2009.13645

The simulation uses the boosted-frame technique as described in
https://www.osti.gov/servlets/purl/940581
In particular, the effect of space-charge is effectively turned off
by initializing an electron and positron beam on top of each other,
each having half the current of the physical beam.

The script checks that the radiation wavelength and gain length
are the expected ones. The check is performed both in the
lab-frame diagnostics and boosted-frame diagnostics.
"""

import sys

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, e, m_e

# Physical parameters of the test
gamma_bunch = 100.6
Bu = 0.5
lambda_u = 3e-2
k_u = 2 * np.pi / lambda_u
K = e * Bu / (m_e * c * k_u)  # Undulator parameter
gamma_boost = (
    gamma_bunch / (1 + K * K / 2) ** 0.5
)  # Lorentz factor of the ponderomotive frame
beta_boost = (1 - 1.0 / gamma_boost**2) ** 0.5


# Analyze the diagnostics showing quantities in the lab frame
filename = sys.argv[1]
ts_lab = OpenPMDTimeSeries(filename)


# Extract the growth of the peak electric field
def extract_peak_E_lab(iteration):
    """
    Extract the position of the peak electric field
    """
    Ex, info = ts_lab.get_field("E", "x", iteration=iteration)
    Ex_max = abs(Ex).max()
    z_max = info.z[abs(Ex).argmax()]
    return z_max, Ex_max


# Loop through all iterations
# Since the radiation power is proportional to the square of the peak electric field,
# the log of the power is equal to the log of the square of the peak electric field,
# up to an additive constant.
z_lab_peak, E_lab_peak = ts_lab.iterate(extract_peak_E_lab)
log_P_peak = np.log(E_lab_peak**2)

# Pick the iterations between which the growth of the log of the power is linear
# (i.e. the growth of the power is exponential) and fit a line to extract the
# gain length.
i_start = 6
i_end = 23
# Perform linear fit
p = np.polyfit(z_lab_peak[i_start:i_end], log_P_peak[i_start:i_end], 1)
# Extract the gain length
Lg = 1 / p[0]
Lg_expected = 0.22  # Expected gain length from https://arxiv.org/pdf/2009.13645
print(f"Gain length: {Lg}")
assert abs(Lg - Lg_expected) / Lg_expected < 0.15

# Check that the radiation wavelength is the expected one
iteration_check = 14
Ex, info = ts_lab.get_field("E", "x", iteration=iteration_check)
Nz = len(info.z)
fft_E = abs(np.fft.fft(Ex))
lambd = 1.0 / np.fft.fftfreq(Nz, d=info.dz)
lambda_radiation_lab = lambd[fft_E[: Nz // 2].argmax()]
lambda_expected = lambda_u / (2 * gamma_boost**2)
print(f"lambda_radiation_lab: {lambda_radiation_lab}")
print(f"lambda_expected: {lambda_expected}")
assert abs(lambda_radiation_lab - lambda_expected) / lambda_expected < 0.01

# Analyze the diagnostics showing quantities in the boosted frame
ts = OpenPMDTimeSeries("diags/diag_boostedframe")


# Extract the growth of the peak electric field
def extract_peak_E_boost(iteration):
    """
    Extract the peak electric field in a *boosted-frame* snapshot.
    Also return the position of the peak in the lab frame.
    """
    Ex, info = ts.get_field("E", "x", iteration=iteration)
    By, info = ts.get_field("B", "y", iteration=iteration)
    E_lab = gamma_boost * (Ex + c * beta_boost * By)
    E_lab_peak = abs(E_lab).max()
    z_boost_peak = info.z[abs(E_lab).argmax()]
    t_boost_peak = ts.current_t
    z_lab_peak = gamma_boost * (z_boost_peak + beta_boost * c * t_boost_peak)
    return z_lab_peak, E_lab_peak


# Loop through all iterations
z_lab_peak, E_lab_peak = ts.iterate(extract_peak_E_boost)
log_P_peak = np.log(E_lab_peak**2)

# Pick the iterations between which the growth of the log of the power is linear
# (i.e. the growth of the power is exponential) and fit a line to extract the
# gain length.
i_start = 16
i_end = 25
# Perform linear fit
p = np.polyfit(z_lab_peak[i_start:i_end], log_P_peak[i_start:i_end], 1)
# Extract the gain length
Lg = 1 / p[0]
Lg_expected = 0.22  # Expected gain length from https://arxiv.org/pdf/2009.13645
print(f"Gain length: {Lg}")
assert abs(Lg - Lg_expected) / Lg_expected < 0.15

# Check that the radiation wavelength is the expected one
iteration_check = 2000
Ex, info = ts.get_field("E", "x", iteration=iteration_check)
By, info = ts.get_field("B", "y", iteration=iteration_check)
E_lab = gamma_boost * (Ex + c * beta_boost * By)
Nz = len(info.z)
fft_E = abs(np.fft.fft(E_lab))
lambd = 1.0 / np.fft.fftfreq(Nz, d=info.dz)
lambda_radiation_boost = lambd[fft_E[: Nz // 2].argmax()]
lambda_radiation_lab = lambda_radiation_boost / (2 * gamma_boost)
lambda_expected = lambda_u / (2 * gamma_boost**2)
assert abs(lambda_radiation_lab - lambda_expected) / lambda_expected < 0.01
