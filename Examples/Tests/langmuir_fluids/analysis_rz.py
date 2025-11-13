#!/usr/bin/env python3

# Copyright 2019 David Grote, Maxence Thevenet, Grant Johnson, Remi Lehe
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from
# the script `inputs.multi.rz.rt`. This simulates a RZ periodic plasma wave.
# The electric field in the simulation is given (in theory) by:
# $$ E_r = -\partial_r \phi = \epsilon \,\frac{mc^2}{e}\frac{2\,r}{w_0^2} \exp\left(-\frac{r^2}{w_0^2}\right) \sin(k_0 z) \sin(\omega_p t)
# $$ E_z = -\partial_z \phi = - \epsilon \,\frac{mc^2}{e} k_0 \exp\left(-\frac{r^2}{w_0^2}\right) \cos(k_0 z) \sin(\omega_p t)
# Unrelated to the Langmuir waves, we also test the plotfile particle filter function in this
# analysis script.
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt

yt.funcs.mylog.setLevel(50)

import numpy as np
from scipy.constants import c, e, epsilon_0, m_e

# this will be the name of the plot file
fn = sys.argv[1]

test_name = os.path.split(os.getcwd())[1]

# Parameters (these parameters must match the parameters in `inputs.multi.rz.rt`)
epsilon = 0.01
n = 2.0e24
w0 = 5.0e-6
n_osc_z = 2
rmin = 0e-6
rmax = 20.0e-6
Nr = 64
zmin = -20e-6
zmax = 20.0e-6
Nz = 128

# Wave vector of the wave
k0 = 2.0 * np.pi * n_osc_z / (zmax - zmin)
# Plasma frequency
wp = np.sqrt((n * e**2) / (m_e * epsilon_0))
kp = wp / c


def Er(z, r, epsilon, k0, w0, wp, t):
    """
    Return the radial electric field as an array
    of the same length as z and r, in the half-plane theta=0
    """
    Er_array = (
        epsilon
        * m_e
        * c**2
        / e
        * 2
        * r
        / w0**2
        * np.exp(-(r**2) / w0**2)
        * np.sin(k0 * z)
        * np.sin(wp * t)
    )
    return Er_array


def Ez(z, r, epsilon, k0, w0, wp, t):
    """
    Return the longitudinal electric field as an array
    of the same length as z and r, in the half-plane theta=0
    """
    Ez_array = (
        -epsilon
        * m_e
        * c**2
        / e
        * k0
        * np.exp(-(r**2) / w0**2)
        * np.cos(k0 * z)
        * np.sin(wp * t)
    )
    return Ez_array


def Jr(z, r, epsilon, k0, w0, wp, t):
    """
    Return the radial current density as an array
    of the same length as z and r, in the half-plane theta=0
    """
    dt = t / 80  # SPECIFIC to config parameters!
    Jr_array = (
        -epsilon_0
        * epsilon
        * m_e
        * c**2
        / e
        * 2
        * r
        / w0**2
        * np.exp(-(r**2) / w0**2)
        * np.sin(k0 * z)
        * np.cos(wp * t - wp * dt / 2)
        * wp
    )  # phase_error = wp*dt/2
    return Jr_array


def Jz(z, r, epsilon, k0, w0, wp, t):
    """
    Return the longitudinal current density as an array
    of the same length as z and r, in the half-plane theta=0
    """
    dt = t / 80  # SPECIFIC to config parameters!
    Jz_array = (
        epsilon_0
        * epsilon
        * m_e
        * c**2
        / e
        * k0
        * np.exp(-(r**2) / w0**2)
        * np.cos(k0 * z)
        * np.cos(wp * t - wp * dt / 2)
        * wp
    )  # phase_error = wp*dt/2
    return Jz_array


def rho(z, r, epsilon, k0, w0, wp, t):
    """
    Return the charge density as an array
    of the same length as z and r, in the half-plane theta=0
    """
    rho_array = (
        epsilon_0
        * epsilon
        * m_e
        * c**2
        / e
        * np.sin(wp * t)
        * np.sin(k0 * z)
        * np.exp(-(r**2) / w0**2)
        * ((4.0 / (w0**2)) * (1 - (r**2) / (w0**2)) + k0**2)
    )
    return rho_array


# Read the file
ds = yt.load(fn)
t0 = ds.current_time.to_value()
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

# Get cell centered coordinates
dr = (rmax - rmin) / Nr
dz = (zmax - zmin) / Nz
coords = np.indices([Nr, Nz], "d")
rr = rmin + (coords[0] + 0.5) * dr
zz = zmin + (coords[1] + 0.5) * dz

# Check the validity of the fields
overall_max_error = 0
Er_sim = data[("boxlib", "Er")].to_ndarray()[:, :, 0]
Er_th = Er(zz, rr, epsilon, k0, w0, wp, t0)
max_error = abs(Er_sim - Er_th).max() / abs(Er_th).max()
print("Er: Max error: %.2e" % (max_error))
overall_max_error = max(overall_max_error, max_error)

Ez_sim = data[("boxlib", "Ez")].to_ndarray()[:, :, 0]
Ez_th = Ez(zz, rr, epsilon, k0, w0, wp, t0)
max_error = abs(Ez_sim - Ez_th).max() / abs(Ez_th).max()
print("Ez: Max error: %.2e" % (max_error))
overall_max_error = max(overall_max_error, max_error)

Jr_sim = data[("boxlib", "jr")].to_ndarray()[:, :, 0]
Jr_th = Jr(zz, rr, epsilon, k0, w0, wp, t0)
max_error = abs(Jr_sim - Jr_th).max() / abs(Jr_th).max()
print("Jr: Max error: %.2e" % (max_error))
overall_max_error = max(overall_max_error, max_error)

Jz_sim = data[("boxlib", "jz")].to_ndarray()[:, :, 0]
Jz_th = Jz(zz, rr, epsilon, k0, w0, wp, t0)
max_error = abs(Jz_sim - Jz_th).max() / abs(Jz_th).max()
print("Jz: Max error: %.2e" % (max_error))
overall_max_error = max(overall_max_error, max_error)

rho_sim = data[("boxlib", "rho")].to_ndarray()[:, :, 0]
rho_th = rho(zz, rr, epsilon, k0, w0, wp, t0)
max_error = abs(rho_sim - rho_th).max() / abs(rho_th).max()
print("rho: Max error: %.2e" % (max_error))
overall_max_error = max(overall_max_error, max_error)

# Plot the last field from the loop (Ez at iteration 40)
plt.subplot2grid((1, 2), (0, 0))
plt.imshow(Ez_sim)
plt.colorbar()
plt.title("Ez, last iteration\n(simulation)")
plt.subplot2grid((1, 2), (0, 1))
plt.imshow(Ez_th)
plt.colorbar()
plt.title("Ez, last iteration\n(theory)")
plt.tight_layout()
plt.savefig(test_name + "_analysis.png")

error_rel = overall_max_error

tolerance_rel = 0.08

print("error_rel    : " + str(error_rel))
print("tolerance_rel: " + str(tolerance_rel))

assert error_rel < tolerance_rel
