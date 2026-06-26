#!/usr/bin/env python3

# Copyright 2020 Andrew Myers, Axel Huebl, Luca Fedeli
# Remi Lehe, Ilian Kara-Mostefa
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


# This file is part of the WarpX automated test suite. It is used to test the
# injection of a laser pulse from an external lasy file:
# - Generate an input lasy file with a gaussian laser pulse.

from lasy.laser import Laser
from lasy.profiles import GaussianProfile

# Physical parameters
um = 1.0e-6
fs = 1.0e-15

# Parameters of the Gaussian beam
wavelength = 1.0 * um
pol = (1, 0)
laser_energy = 1.0
w0 = 12.0 * um
tt = 10.0 * fs

# Create a laser using lasy
profile = GaussianProfile(wavelength, pol, laser_energy, w0, tt, t_peak=0)
dim = "xyt"
lo = (-25e-6, -25e-6, -20e-15)
hi = (+25e-6, +25e-6, +20e-15)
npoints = (100, 100, 100)
laser = Laser(dim, lo, hi, npoints, profile)
laser.normalize(laser_energy, kind="energy")
laser.write_to_file("gaussian_laser_3d")
