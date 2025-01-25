#!/usr/bin/env python3
#
# Copyright 2024 Remi Lehe
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
This script tests the emission of particles from the embedded boundary.
(In this case, the embedded boundary is a sphere in 3D and RZ, a cylinder in 2D.)
We check that the embedded boundary emits the correct number of particles, and that
the particle distributions are consistent with the expected distributions.
"""

import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import yt
from scipy.constants import c, m_e
from scipy.special import erf

yt.funcs.mylog.setLevel(0)

# Open plotfile specified in command line
fn = sys.argv[1]
ds = yt.load(fn)
ad = ds.all_data()
t_inj = 0.5e-8  # duration for which the flux injection was active

# Extract the dimensionality of the simulation
with open("./warpx_used_inputs", "r") as f:
    warpx_used_inputs = f.read()
if re.search("geometry.dims = 2", warpx_used_inputs):
    dims = "2D"
elif re.search("geometry.dims = RZ", warpx_used_inputs):
    dims = "RZ"
elif re.search("geometry.dims = 3", warpx_used_inputs):
    dims = "3D"

# Total number of electrons expected:
# Simulation parameters determine the total number of particles emitted (Ntot)
flux = 1.0  # in m^-2.s^-1, from the input script
R = 2.0  # in m, radius of the sphere
if dims == "3D" or dims == "RZ":
    emission_surface = 4 * np.pi * R**2  # in m^2
elif dims == "2D":
    emission_surface = 2 * np.pi * R  # in m
Ntot = flux * emission_surface * t_inj

# Parameters of the histogram
hist_bins = 50
hist_range = [-0.5, 0.5]


# Define function that histograms and checks the data
def gaussian_dist(u, u_th):
    return 1.0 / ((2 * np.pi) ** 0.5 * u_th) * np.exp(-(u**2) / (2 * u_th**2))


def gaussian_flux_dist(u, u_th, u_m):
    normalization_factor = u_th**2 * np.exp(-(u_m**2) / (2 * u_th**2)) + (
        np.pi / 2
    ) ** 0.5 * u_m * u_th * (1 + erf(u_m / (2**0.5 * u_th)))
    result = (
        1.0
        / normalization_factor
        * np.where(u > 0, u * np.exp(-((u - u_m) ** 2) / (2 * u_th**2)), 0)
    )
    return result


def compare_gaussian(u, w, u_th, label=""):
    du = (hist_range[1] - hist_range[0]) / hist_bins
    w_hist, u_hist = np.histogram(u, bins=hist_bins, weights=w / du, range=hist_range)
    u_hist = 0.5 * (u_hist[1:] + u_hist[:-1])
    w_th = Ntot * gaussian_dist(u_hist, u_th)
    plt.plot(u_hist, w_hist, label=label + ": simulation")
    plt.plot(u_hist, w_th, "--", label=label + ": theory")
    assert np.allclose(w_hist, w_th, atol=0.07 * w_th.max())


def compare_gaussian_flux(u, w, u_th, u_m, label=""):
    du = (hist_range[1] - hist_range[0]) / hist_bins
    w_hist, u_hist = np.histogram(u, bins=hist_bins, weights=w / du, range=hist_range)
    u_hist = 0.5 * (u_hist[1:] + u_hist[:-1])
    w_th = Ntot * gaussian_flux_dist(u_hist, u_th, u_m)
    plt.plot(u_hist, w_hist, label=label + ": simulation")
    plt.plot(u_hist, w_th, "--", label=label + ": theory")
    assert np.allclose(w_hist, w_th, atol=0.05 * w_th.max())


# Load data and perform check

plt.figure()

if dims == "3D":
    x = ad["electron", "particle_position_x"].to_ndarray()
    y = ad["electron", "particle_position_y"].to_ndarray()
    z = ad["electron", "particle_position_z"].to_ndarray()
elif dims == "2D":
    x = ad["electron", "particle_position_x"].to_ndarray()
    y = np.zeros_like(x)
    z = ad["electron", "particle_position_y"].to_ndarray()
elif dims == "RZ":
    theta = ad["electron", "particle_theta"].to_ndarray()
    r = ad["electron", "particle_position_x"].to_ndarray()
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = ad["electron", "particle_position_y"].to_ndarray()
ux = ad["electron", "particle_momentum_x"].to_ndarray() / (m_e * c)
uy = ad["electron", "particle_momentum_y"].to_ndarray() / (m_e * c)
uz = ad["electron", "particle_momentum_z"].to_ndarray() / (m_e * c)
w = ad["electron", "particle_weight"].to_ndarray()

# Check that the total number of particles emitted is correct
Ntot_sim = np.sum(w)
print("Ntot_sim = ", Ntot_sim)
print("Ntot = ", Ntot)
assert np.isclose(Ntot_sim, Ntot, rtol=0.01)

# Check that none of the particles are inside the EB
# A factor 0.98 is applied to accomodate
# the cut-cell approximation of the sphere
assert np.all(x**2 + y**2 + z**2 > (0.98 * R) ** 2)

# Check that the normal component of the velocity is consistent with the expected distribution
r = np.sqrt(x**2 + y**2 + z**2)
nx = x / r
ny = y / r
nz = z / r
u_n = ux * nx + uy * ny + uz * nz  # normal component
compare_gaussian_flux(u_n, w, u_th=0.1, u_m=0.07, label="u_n")

# Pick a direction that is orthogonal to the normal direction, and check the distribution
vx = ny / np.sqrt(nx**2 + ny**2)
vy = -nx / np.sqrt(nx**2 + ny**2)
vz = 0
u_perp = ux * vx + uy * vy + uz * vz
compare_gaussian(u_perp, w, u_th=0.01, label="u_perp")

# Pick the other perpendicular direction, and check the distribution
# The third direction is obtained by the cross product (n x v)
wx = ny * vz - nz * vy
wy = nz * vx - nx * vz
wz = nx * vy - ny * vx
u_perp2 = ux * wx + uy * wy + uz * wz
compare_gaussian(u_perp2, w, u_th=0.01, label="u_perp")

plt.tight_layout()
plt.savefig("Distribution.png")
