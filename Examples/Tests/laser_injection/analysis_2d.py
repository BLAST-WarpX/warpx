#!/usr/bin/env python3

# Copyright 2019 Andrew Myers, Jean-Luc Vay, Maxence Thevenet
# Remi Lehe, Weiqun Zhang, Luca Fedeli
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This file is part of the WarpX automated test suite. Its purpose is to test the
# injection of a Gaussian laser pulse from an antenna in a 2D simulation.
# In order to avoid privileged directions, the laser is injected at
# approximately 27 degrees with respect to the x axis. Moreover the polarization axis is neither
# parallel nor perpendicular to the xz plane. Finally moving window along the
# x axis is enabled.
# The test calculates the envelope of each component of the laser pulse at the end of
# the simulation and it compares it with theory. It also checks that the
# central frequency of the Fourier transform is the expected one.

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import hilbert

# Maximum acceptable error for this test
relative_error_threshold = 0.05

# A small number
small_num = 1.0e-8

# Physical parameters
um = 1.0e-6
fs = 1.0e-15
c = 299792458

# Parameters of the gaussian beam
wavelength = 1.0 * um
w0 = 5.0 * um
tt = 10.0 * fs
x_c = 10.0 * um
t_c = 24.0 * fs
# foc_dist = 13.109*um (not actually used)
E_max = 4e12

# laser direction
dir_vector = np.array([2.0, 0, 1.0])
dir_vector /= np.linalg.norm(dir_vector)

rot_angle = np.arctan(dir_vector[2] / dir_vector[0])

# polarization vector
pol_vector = np.array([1.0, 1.0, -2.0])
pol_vector /= np.linalg.norm(pol_vector)


# Calculates the envelope of a Gaussian beam
def gauss_env(T, XX, ZZ):
    """Function to compute the theory for the envelope"""

    Z = np.cos(rot_angle) * (XX - x_c) + np.sin(rot_angle) * ZZ
    X = -np.sin(rot_angle) * (XX - x_c) + np.cos(rot_angle) * ZZ

    inv_tau2 = 1.0 / tt / tt
    inv_w_2 = 1.0 / (w0 * w0)
    exp_arg = -(X * X) * inv_w_2 - inv_tau2 / c / c * (Z - T * c) * (Z - T * c)
    return E_max * np.real(np.exp(exp_arg))


# Checks envelope and central frequency for a given laser component
def check_component(data, component, t_env_theory, coeff, X, Z, dx, dz):
    print("*** Checking " + component + " ***")
    field = data["boxlib", component].v.squeeze()
    env = abs(hilbert(field))

    env_theory = t_env_theory * np.abs(coeff)

    # Plot results
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(221, aspect="equal")
    ax1.set_title("PIC field")
    p1 = ax1.pcolormesh(X, Z, field)
    cax1 = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(p1, cax=cax1, orientation="vertical")

    ax2 = fig.add_subplot(222, aspect="equal")
    ax2.set_title("PIC envelope")
    p2 = ax2.pcolormesh(X, Z, env)
    cax2 = make_axes_locatable(ax2).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(p2, cax=cax2, orientation="vertical")

    ax3 = fig.add_subplot(223, aspect="equal")
    ax3.set_title("Theory envelope")
    p3 = ax3.pcolormesh(X, Z, env_theory)
    cax3 = make_axes_locatable(ax3).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(p3, cax=cax3, orientation="vertical")

    ax4 = fig.add_subplot(224, aspect="equal")
    ax4.set_title("Difference")
    p4 = ax4.pcolormesh(X, Z, env - env_theory)
    cax4 = make_axes_locatable(ax4).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(p4, cax=cax4, orientation="vertical")

    plt.tight_layout()
    plt.savefig("plt_" + component + ".png", bbox_inches="tight")

    if np.abs(coeff) < small_num:
        is_field_zero = np.sum(np.abs(env)) < small_num
        if is_field_zero:
            print("[OK] Field component expected to be 0 is ~ 0")
        else:
            print("[FAIL] Field component expected to be 0 is NOT ~ 0")
        assert is_field_zero
        print("******\n")
        return

    relative_error_env = np.sum(np.abs(env - env_theory)) / np.sum(np.abs(env_theory))
    is_env_ok = relative_error_env < relative_error_threshold
    if is_env_ok:
        print(
            "[OK] Relative error envelope: {:6.3f} %".format(relative_error_env * 100)
        )
    else:
        print(
            "[FAIL] Relative error envelope: {:6.3f} %".format(relative_error_env * 100)
        )
    assert is_env_ok

    fft_field = np.fft.fft2(field)

    freq_rows = np.fft.fftfreq(fft_field.shape[0], dx / c)
    freq_cols = np.fft.fftfreq(fft_field.shape[1], dz / c)

    pos_max = np.unravel_index(np.abs(fft_field).argmax(), fft_field.shape)

    freq = np.sqrt((freq_rows[pos_max[0]]) ** 2 + (freq_cols[pos_max[1]] ** 2))
    exp_freq = c / wavelength

    relative_error_freq = np.abs(freq - exp_freq) / exp_freq
    is_freq_ok = relative_error_freq < relative_error_threshold
    if is_freq_ok:
        print(
            "[OK] Relative error frequency: {:6.3f} %".format(relative_error_freq * 100)
        )
    else:
        print(
            "[FAIL] Relative error frequency: {:6.3f} %".format(
                relative_error_freq * 100
            )
        )
    assert is_freq_ok

    print("******\n")


def check_laser(filename):
    ds = yt.load(filename)

    # yt 4.0+ has rounding issues with our domain data:
    # RuntimeError: yt attempted to read outside the boundaries
    # of a non-periodic domain along dimension 0.
    if "force_periodicity" in dir(ds):
        ds.force_periodicity()

    x = np.linspace(
        ds.domain_left_edge[0].v, ds.domain_right_edge[0].v, ds.domain_dimensions[0]
    )

    dx = (ds.domain_right_edge[0].v - ds.domain_left_edge[0].v) / (
        ds.domain_dimensions[0] - 1
    )

    z = np.linspace(
        ds.domain_left_edge[1].v, ds.domain_right_edge[1].v, ds.domain_dimensions[1]
    )

    dz = (ds.domain_right_edge[1].v - ds.domain_left_edge[1].v) / (
        ds.domain_dimensions[1] - 1
    )

    X, Z = np.meshgrid(x, z, indexing="ij")

    # Compute the theory for envelope
    env_theory = gauss_env(+t_c - ds.current_time.to_value(), X, Z) + gauss_env(
        -t_c + ds.current_time.to_value(), X, Z
    )

    # Read laser field in PIC simulation, and compute envelope
    all_data_level_0 = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )

    b_vector = np.cross(dir_vector, pol_vector)

    components = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
    coeffs = [
        pol_vector[0],
        pol_vector[1],
        pol_vector[2],
        b_vector[0],
        b_vector[1],
        b_vector[2],
    ]

    field_facts = [1, 1, 1, 1 / c, 1 / c, 1 / c]

    for comp, coeff, field_fact in zip(components, coeffs, field_facts):
        check_component(
            all_data_level_0, comp, field_fact * env_theory, coeff, X, Z, dx, dz
        )


def main():
    filename_end = sys.argv[1]

    check_laser(filename_end)


if __name__ == "__main__":
    main()
