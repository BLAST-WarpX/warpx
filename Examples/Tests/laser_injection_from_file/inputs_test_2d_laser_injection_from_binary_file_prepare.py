#!/usr/bin/env python3

# Copyright 2020 Andrew Myers, Axel Huebl, Luca Fedeli
# Remi Lehe, Ilian Kara-Mostefa
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


# This file is part of the WarpX automated test suite. It is used to test the
# injection of a laser pulse from an external binary file:
# - Generate an input binary file with a gaussian laser pulse.

import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import yt

yt.funcs.mylog.setLevel(50)

sys.path.insert(1, "../../../../warpx/Regression/Checksum/")

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


def gauss(T, X, Y, opt):
    """Compute the electric field for a Gaussian laser pulse.
    This is used to write the binary input file.
    """

    k0 = 2.0 * np.pi / wavelength
    inv_tau2 = 1.0 / tt / tt
    osc_phase = k0 * c * (T - t_c)

    diff_factor = 1.0 + 1.0j * foc_dist * 2 / (k0 * w0 * w0)
    inv_w_2 = 1.0 / (w0 * w0 * diff_factor)

    pre_fact = np.exp(1.0j * osc_phase)

    if opt == "3d":
        pre_fact = pre_fact / diff_factor
    else:
        pre_fact = pre_fact / np.sqrt(diff_factor)

    exp_arg = -(X * X + Y * Y) * inv_w_2 - inv_tau2 * (T - t_c) * (T - t_c)

    return np.real(pre_fact * np.exp(exp_arg))


def write_file(fname, x, y, t, E):
    """For a given filename fname, space coordinates x and y, time coordinate t
    and field E, write a WarpX-compatible input binary file containing the
    profile of the laser pulse. This function should be used in the case
    of a uniform spatio-temporal mesh
    """

    with open(fname, "wb") as file:
        flag_unif = 1
        file.write(flag_unif.to_bytes(1, byteorder="little"))
        file.write((len(t)).to_bytes(4, byteorder="little", signed=False))
        file.write((len(x)).to_bytes(4, byteorder="little", signed=False))
        file.write((len(y)).to_bytes(4, byteorder="little", signed=False))
        file.write(t[0].tobytes())
        file.write(t[-1].tobytes())
        file.write(x[0].tobytes())
        file.write(x[-1].tobytes())
        if len(y) == 1:
            file.write(y[0].tobytes())
        else:
            file.write(y[0].tobytes())
            file.write(y[-1].tobytes())
        file.write(E.tobytes())


T, X, Y = np.meshgrid(tcoords, xcoords, np.array([0.0]), indexing="ij")
E_t = gauss(T, X, Y, "2d")
write_file("gauss_2d", xcoords, np.array([0.0]), tcoords, E_t)
