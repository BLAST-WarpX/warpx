#!/usr/bin/env python3

# Copyright 2019-2020 Yinjian Zhao
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This script tests initial distributions.
# 1 denotes gaussian distribution.
# 2 denotes maxwell-boltzmann distribution.
# 3 denotes maxwell-juttner distribution.
# 4 denotes gaussian position distribution.
# 5 denotes maxwell-juttner distribution w/ spatially varying temperature
# 6 denotes maxwell-boltzmann distribution w/ constant velocity
# 7 denotes maxwell-boltzmann distribution w/ spatially-varying velocity
# 8 denotes uniform distribution
# 9 denotes gaussian_parser distribution w/ spatially-varying mean and thermal velocity
# The distribution is obtained through reduced diagnostic ParticleHistogram.

import numpy as np
import scipy.constants as scc
import scipy.special as scs
from read_raw_data import read_reduced_diags, read_reduced_diags_histogram

# print tolerance
tolerance = 0.02
print("Tolerance:", tolerance)

# ===============================
# gaussian and maxwell-boltzmann
# ===============================

# load data
bin_value, h1x = read_reduced_diags_histogram("h1x.txt")[2:]
h1y = read_reduced_diags_histogram("h1y.txt")[3]
h1z = read_reduced_diags_histogram("h1z.txt")[3]
h2x = read_reduced_diags_histogram("h2x.txt")[3]
h2y = read_reduced_diags_histogram("h2y.txt")[3]
h2z = read_reduced_diags_histogram("h2z.txt")[3]

# parameters of theory
u_rms = 0.01
gamma = np.sqrt(1.0 + u_rms * u_rms)
v_rms = u_rms / gamma * scc.c
n = 1.0e21
V = 8.0
db = 0.0016

# compute the analytical solution
f = (
    n
    * V
    * scc.c
    * db
    * np.exp(-0.5 * (bin_value * scc.c / v_rms) ** 2)
    / (v_rms * np.sqrt(2.0 * scc.pi))
)
f_peak = np.amax(f)

# compute error
# note that parameters are chosen such that gaussian and
# maxwell-boltzmann distributions are identical
f1_error = (
    np.sum(np.abs(f - h1x) + np.abs(f - h1y) + np.abs(f - h1z))
    / bin_value.size
    / f_peak
)
f2_error = (
    np.sum(np.abs(f - h2x) + np.abs(f - h2y) + np.abs(f - h2z))
    / bin_value.size
    / f_peak
)

print("Gaussian distribution difference:", f1_error)
print("Maxwell-Boltzmann distribution difference:", f2_error)

assert f1_error < tolerance
assert f2_error < tolerance

# ================
# maxwell-juttner
# ================

# load data
bin_value, bin_data = read_reduced_diags_histogram("h3.txt")[2:]
bin_data_filtered = read_reduced_diags_histogram("h3_filtered.txt")[3]

# parameters of theory
theta = 1.0
K2 = scs.kn(2, 1.0 / theta)
n = 1.0e21
V = 8.0
db = 0.22

# compute the analytical solution
f = (
    n
    * V
    * db
    * bin_value**2
    * np.sqrt(1.0 - 1.0 / bin_value**2)
    / (theta * K2)
    * np.exp(-bin_value / theta)
)
f_peak = np.amax(f)

# analytical solution for the filtered histogram: we just filter out gamma values < 5.5
f_filtered = f * (bin_value > 5.5)

# compute error
f3_error = (
    np.sum(np.abs(f - bin_data) + np.abs(f_filtered - bin_data_filtered))
    / bin_value.size
    / f_peak
)

print("Maxwell-Juttner distribution difference:", f3_error)

assert f3_error < tolerance

# ==============
# gaussian beam
# ==============

# load data
bin_value, h4x = read_reduced_diags_histogram("h4x.txt")[2:]
h4y = read_reduced_diags_histogram("h4y.txt")[3]
h4z = read_reduced_diags_histogram("h4z.txt")[3]
_, bmmntr = read_reduced_diags("bmmntr.txt")
charge = bmmntr["charge"][0]

# parameters of theory
x_rms = 0.25
z_cut = 2.0
q_tot = -1.0e-20
q_e = -1.602176634e-19
npart = q_tot / q_e
db = bin_value[1] - bin_value[0]

# compute the analytical solution
f_xy = (
    npart
    * db
    * np.exp(-0.5 * (bin_value / x_rms) ** 2)
    / (x_rms * np.sqrt(2.0 * scc.pi))
    * scs.erf(z_cut / np.sqrt(2.0))
)
f_z = (
    npart
    * db
    * np.exp(-0.5 * (bin_value / x_rms) ** 2)
    / (x_rms * np.sqrt(2.0 * scc.pi))
)
f_z[np.absolute(bin_value) > z_cut * x_rms] = 0.0
f_peak = np.amax(f_z)
q_tot_cut = q_tot * scs.erf(z_cut / np.sqrt(2.0))

# compute error
f4_error = (
    np.sum(np.abs(f_xy - h4x) + np.abs(f_xy - h4y) + np.abs(f_z - h4z))
    / bin_value.size
    / f_peak
)
charge_error = np.abs((q_tot_cut - charge) / q_tot)

do_plot = False
if do_plot:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(121)
    plt.plot(bin_value, f_xy, "+-", label="ref")
    plt.plot(bin_value, h4x, "+--", label="sim")
    plt.legend()
    plt.subplot(122)
    plt.plot(bin_value, f_z, "+-", label="ref")
    plt.plot(bin_value, h4z, "+--", label="sim")
    plt.legend()
    plt.savefig("toto.pdf", bbox_inches="tight")

print("Gaussian position distribution difference:", f4_error)
assert f4_error < tolerance

print("Relative beam charge difference:", charge_error)
assert charge_error < tolerance

# =============================================
# maxwell-juttner with temperature from parser
# =============================================

# load data
bin_value, bin_data_neg = read_reduced_diags_histogram("h5_neg.txt")[2:]
bin_data_pos = read_reduced_diags_histogram("h5_pos.txt")[3]

# parameters of theory
# _neg denotes where x<0, _pos where x>0
theta_neg = 1.0
theta_pos = 2.0
K2_neg = scs.kn(2, 1.0 / theta_neg)
K2_pos = scs.kn(2, 1.0 / theta_pos)
n = 1.0e21
V = 8.0 / 2  # because each of these are for half the domain
db = 0.22

# compute the analytical solution for each half of the domain
f_neg = (
    n
    * V
    * db
    * bin_value**2
    * np.sqrt(1.0 - 1.0 / bin_value**2)
    / (theta_neg * K2_neg)
    * np.exp(-bin_value / theta_neg)
)
f_neg_peak = np.amax(f_neg)
f_pos = (
    n
    * V
    * db
    * bin_value**2
    * np.sqrt(1.0 - 1.0 / bin_value**2)
    / (theta_pos * K2_pos)
    * np.exp(-bin_value / theta_pos)
)
f_pos_peak = np.amax(f_pos)
f_peak = max(f_neg_peak, f_pos_peak)

# compute error
f5_error = (
    np.sum(np.abs(f_neg - bin_data_neg) + np.abs(f_pos - bin_data_pos))
    / bin_value.size
    / f_peak
)

print("Maxwell-Juttner parser temperature difference:", f5_error)

assert f5_error < tolerance

# ==============================================
# maxwell-boltzmann with constant bulk velocity
# ==============================================

# load data
bin_value_g, bin_data_g = read_reduced_diags_histogram("h6.txt")[2:]
bin_value_uy, bin_data_uy = read_reduced_diags_histogram("h6uy.txt")[2:]

# Expected values for beta and u = beta*gamma
beta_const = 0.2
g_const = 1.0 / np.sqrt(1.0 - beta_const * beta_const)
uy_const = beta_const * g_const
g_bin_size = 0.004
g_bin_min = 1.0
uy_bin_size = 0.04
uy_bin_min = -1.0
V = 8.0  # volume in m^3
n = 1.0e21  # number density in 1/m^3

f_g = np.zeros_like(bin_value_g)
i_g = int(np.floor((g_const - g_bin_min) / g_bin_size))
f_g[i_g] = n * V
f_peak = np.amax(f_g)

f_uy = np.zeros_like(bin_value_uy)
i_uy = int(np.floor((-uy_const - uy_bin_min) / uy_bin_size))
f_uy[i_uy] = n * V

f6_error = (
    np.sum(np.abs(f_g - bin_data_g) + np.abs(f_uy - bin_data_uy))
    / bin_value_g.size
    / f_peak
)

print("Maxwell-Boltzmann constant velocity difference:", f6_error)

assert f6_error < tolerance

# ============================================
# maxwell-boltzmann with parser bulk velocity
# ============================================

# load data
bin_value_g, bin_data_g = read_reduced_diags_histogram("h7.txt")[2:]
bin_value_uy, bin_data_uy_neg = read_reduced_diags_histogram("h7uy_neg.txt")[2:]
bin_data_uy_pos = read_reduced_diags_histogram("h7uy_pos.txt")[3]

# Expected values for beta and u = beta*gamma
beta_const = 0.2
g_const = 1.0 / np.sqrt(1.0 - beta_const * beta_const)
uy_const = beta_const * g_const
g_bin_size = 0.004
g_bin_min = 1.0
uy_bin_size = 0.04
uy_bin_min = -1.0
V = 8.0  # volume in m^3
n = 1.0e21  # number density in 1/m^3

f_g = np.zeros_like(bin_value_g)
i_g = int(np.floor((g_const - g_bin_min) / g_bin_size))
f_g[i_g] = n * V
f_peak = np.amax(f_g)

f_uy_neg = np.zeros_like(bin_value_uy)
i_uy_neg = int(np.floor((uy_const - uy_bin_min) / uy_bin_size))
f_uy_neg[i_uy_neg] = n * V / 2.0

f_uy_pos = np.zeros_like(bin_value_uy)
i_uy_pos = int(np.floor((-uy_const - uy_bin_min) / uy_bin_size))
f_uy_pos[i_uy_pos] = n * V / 2.0

f7_error = (
    np.sum(
        np.abs(f_g - bin_data_g)
        + np.abs(f_uy_pos - bin_data_uy_pos)
        + np.abs(f_uy_neg - bin_data_uy_neg)
    )
    / bin_value_g.size
    / f_peak
)

print("Maxwell-Boltzmann parser velocity difference:", f7_error)

assert f7_error < tolerance


# ============================================
# Cuboid distribution in momentum space
# ============================================

bin_value_x, h8x = read_reduced_diags_histogram("h8x.txt")[2:]
bin_value_y, h8y = read_reduced_diags_histogram("h8y.txt")[2:]
bin_value_z, h8z = read_reduced_diags_histogram("h8z.txt")[2:]

# Analytical distribution
ux_min = -0.2
ux_max = 0.3
uy_min = -0.1
uy_max = 0.1
uz_min = 10
uz_max = 11.2

N0 = n * V

# Distributions along the three momentum axes are independent:
# we can test them separately


# This counts the number of bins where we expect the distribution to be nonzero
def nonzero_bins(bins, low, high):
    # Bin with nonzero distribution is defined when b_{i+1} > u_min & b_i < u_max
    # `bins` contains the bin centers

    db = bins[1] - bins[0]
    loweredges = bins - 0.5 * db
    upperedges = bins + 0.5 * db
    return (upperedges > low) & (loweredges < high)


# Function that checks the validity of the histogram.
# We have to call it for each of the axis
def check_validity_uniform(bins, histogram, u_min, u_max, Ntrials=1000):
    """
    - `bins` contains the bin centers
    - `histogram` contains the normalized histogram (i.e. np.sum(histogram) = 1)
    - `u_min` is the minimum of the histogram domain
    - `u_max` is the maximum of the histogram domain
    """
    nzbins = nonzero_bins(bins, u_min, u_max)
    Nbins = np.count_nonzero(nzbins)
    db = bins[1] - bins[0]
    loweredges = bins - 0.5 * db
    upperedges = bins + 0.5 * db

    # First we check if Nbins = 1 because this covers the case
    # u_max = u_min (i.e. a delta distribution)
    if Nbins == 1:
        # In this case the result should be exact
        assert (histogram[nzbins].item() - 1) < 1e-8

        return

    # The probability of filling a given bin is proportional to the bin width.
    # We normalize it to the "full bin" value (i.e. every bin except from the edges
    # is expected to have the same p in a uniform distribution).
    # The fill ratio is therefore 1 for a bin fully included in the domain and < 1 else.
    # Filling a given bin is a binomial process, so we basically test each histogram with the
    # expected average value to be (x - mu) < 3 sigma

    probability = (
        np.clip(upperedges, u_min, u_max) - np.clip(loweredges, u_min, u_max)
    ) / (u_max - u_min)
    variance = probability * (1 - probability)
    nzprob = probability[nzbins]
    nzhist = histogram[nzbins]
    nzvar = variance[nzbins]
    samplesigma = 1 / np.sqrt(Ntrials)

    normalizedvariable = np.abs(nzhist - nzprob) / np.sqrt(nzvar)

    assert np.all(normalizedvariable < 3 * samplesigma)


# Test the distribution at every time step
# (this assumes that no interaction is happening)
for timestep in range(len(h8x)):
    check_validity_uniform(bin_value_x, h8x[timestep] / N0, ux_min, ux_max)
    check_validity_uniform(bin_value_y, h8y[timestep] / N0, uy_min, uy_max)
    check_validity_uniform(bin_value_z, h8z[timestep] / N0, uz_min, uz_max)

# =================================================
# Gaussian with parser mean and standard deviation
# =================================================

# load data
bin_value_ux, bin_data_ux = read_reduced_diags_histogram("h9x.txt")[2:]
bin_value_uy, bin_data_uy = read_reduced_diags_histogram("h9y.txt")[2:]
bin_value_uz, bin_data_uz = read_reduced_diags_histogram("h9z.txt")[2:]


def Gaussian(mean, sigma, u):
    V = 8.0  # volume in m^3
    n = 1.0e21  # number density in 1/m^3
    return (n * V / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -((u - mean) ** 2) / (2.0 * sigma**2)
    )


du = 2.0 / 50
f_ux = Gaussian(0.1, 0.2, bin_value_ux) * du
f_uy = Gaussian(0.12, 0.21, bin_value_uy) * du
f_uz = Gaussian(0.14, 0.22, bin_value_uz) * du

f9_error = (
    np.sum(
        np.abs(f_ux - bin_data_ux) / f_ux.max()
        + np.abs(f_uy - bin_data_uy) / f_ux.max()
        + np.abs(f_uz - bin_data_uz) / f_uz.max()
    )
    / bin_value_ux.size
)

print("gaussian_parse_momentum_function velocity difference:", f9_error)

assert f9_error < tolerance
