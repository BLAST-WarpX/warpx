#!/usr/bin/env python3

# Copyright 2020 Andrew Myers, Axel Huebl, Luca Fedeli
# Remi Lehe, Ilian Kara-Mostefa
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


# This file is part of the WarpX automated test suite. It is used to test the
# injection of a laser pulse from an external lasy file:
# - Generate an input lasy file with a Laguerre Gaussian laser pulse.

from lasy.laser import Laser
from lasy.profiles import CombinedLongitudinalTransverseProfile
from lasy.profiles.longitudinal import GaussianLongitudinalProfile
from lasy.profiles.transverse import LaguerreGaussianTransverseProfile

# Maximum acceptable error for this test
relative_error_threshold = 0.065

# Physical parameters
um = 1.0e-6
fs = 1.0e-15

# Parameters of the Laguerre Gaussian beam
wavelength = 1.0 * um
pol = (1, 0)
laser_energy = 1.0
w0 = 12.0 * um
tt = 10.0 * fs
t_c = 20.0 * fs

# Create a Laguerre Gaussian laser in RZ geometry using lasy
profile = CombinedLongitudinalTransverseProfile(
    wavelength,
    pol,
    laser_energy,
    GaussianLongitudinalProfile(wavelength, tt, t_peak=0),
    LaguerreGaussianTransverseProfile(w0, p=0, m=1),
)
dim = "rt"
lo = (0e-6, -20e-15)
hi = (+25e-6, +20e-15)
npoints = (100, 100)
laser = Laser(dim, lo, hi, npoints, profile, n_azimuthal_modes=2)
laser.normalize(laser_energy, kind="energy")
laser.write_to_file("laguerre_laser_RZ")
