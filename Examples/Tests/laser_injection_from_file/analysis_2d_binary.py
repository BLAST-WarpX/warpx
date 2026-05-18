#!/usr/bin/env python3

# Copyright 2020 Andrew Myers, Axel Huebl, Luca Fedeli
# Remi Lehe, Ilian Kara-Mostefa
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


# This file is part of the WarpX automated test suite. It is used to test the
# injection of a laser pulse from an external binary file:
# - Compute the theory for laser envelope at time T
# - Compare theory and simulation in 2D, for both envelope and central frequency

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt
from scipy.signal import hilbert

yt.funcs.mylog.setLevel(50)

# Maximum acceptable error for this test
relative_error_threshold = 0.065

# Physical parameters
um = 1.0e-6
fs = 1.0e-15
c = 299792458

# Parameters of the gaussian beam
wavelength = 1.0 * um
w0 = 6.0 * um
tt = 10.0 * fs
x_c = 0.0 * um
t_c = 20.0 * fs
foc_dist = 10 * um
E_max = 1e12
rot_angle = -np.pi / 4.0

# Parameters of the tx grid
x_l = -12.0 * um
x_r = 12.0 * um
x_points = 480
t_l = 0.0 * fs
t_r = 40.0 * fs
t_points = 400
tcoords = np.linspace(t_l, t_r, t_points)
xcoords = np.linspace(x_l, x_r, x_points)


# Function for the envelope
def gauss_env(T, XX, ZZ):
    """Function to compute the theory for the envelope"""

    X = np.cos(rot_angle) * XX + np.sin(rot_angle) * ZZ
    Z = -np.sin(rot_angle) * XX + np.cos(rot_angle) * ZZ

    inv_tau2 = 1.0 / tt / tt
    inv_w_2 = 1.0 / (w0 * w0)
    exp_arg = -(X * X) * inv_w_2 - inv_tau2 / c / c * (Z - T * c) * (Z - T * c)
    return E_max * np.real(np.exp(exp_arg))


filename = sys.argv[1]
compname = "comp_unf.pdf"
steps = 250
ds = yt.load(filename)

dt = ds.current_time.to_value() / steps

# Define 2D meshes
x = np.linspace(
    ds.domain_left_edge[0], ds.domain_right_edge[0], ds.domain_dimensions[0]
).v
z = np.linspace(
    ds.domain_left_edge[ds.dimensionality - 1],
    ds.domain_right_edge[ds.dimensionality - 1],
    ds.domain_dimensions[ds.dimensionality - 1],
).v
X, Z = np.meshgrid(x, z, sparse=False, indexing="ij")

# Compute the theory for envelope
env_theory = gauss_env(+t_c - ds.current_time.to_value(), X, Z) + gauss_env(
    -t_c + ds.current_time.to_value(), X, Z
)

# Read laser field in PIC simulation, and compute envelope
all_data_level_0 = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
F_laser = all_data_level_0["boxlib", "Ey"].v.squeeze()
env = abs(hilbert(F_laser))
extent = [
    ds.domain_left_edge[ds.dimensionality - 1],
    ds.domain_right_edge[ds.dimensionality - 1],
    ds.domain_left_edge[0],
    ds.domain_right_edge[0],
]

# Plot results
plt.figure(figsize=(8, 6))
plt.subplot(221)
plt.title("PIC field")
plt.imshow(F_laser, extent=extent)
plt.colorbar()
plt.subplot(222)
plt.title("PIC envelope")
plt.imshow(env, extent=extent)
plt.colorbar()
plt.subplot(223)
plt.title("Theory envelope")
plt.imshow(env_theory, extent=extent)
plt.colorbar()
plt.subplot(224)
plt.title("Difference")
plt.imshow(env - env_theory, extent=extent)
plt.colorbar()
plt.tight_layout()
plt.savefig(compname, bbox_inches="tight")

relative_error_env = np.sum(np.abs(env - env_theory)) / np.sum(np.abs(env))
print("Relative error envelope: ", relative_error_env)
assert relative_error_env < relative_error_threshold

fft_F_laser = np.fft.fft2(F_laser)

freq_rows = np.fft.fftfreq(F_laser.shape[0], dt)
freq_cols = np.fft.fftfreq(F_laser.shape[1], dt)

pos_max = np.unravel_index(np.abs(fft_F_laser).argmax(), fft_F_laser.shape)

freq = np.sqrt((freq_rows[pos_max[0]]) ** 2 + (freq_cols[pos_max[1]] ** 2))
exp_freq = c / wavelength

relative_error_freq = np.abs(freq - exp_freq) / exp_freq
print("Relative error frequency: ", relative_error_freq)
assert relative_error_freq < relative_error_threshold
