#!/usr/bin/env python3

# Copyright 2025 Remi Lehe
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This tests the loading of the plasma density from a file.
# An openPMD file is created, containing a density corresponding to
# a plasma channel (parabolic in x, y, and linear ramp in z followed by a plateau)
# This file checks that the density of the particles that are injected in the simulation
# (including continuous injection by a moving window) corresponds to the expected density.

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import e

ts = OpenPMDTimeSeries("./diags/diag")

# Loop over all iterations, so that we test loading of the density
# as the moving window moves.
for iteration in ts.iterations:
    rho, info = ts.get_field("rho", iteration=iteration)
    density_sim = -rho / e

    # Compute expected density
    on_axis_density = 1e24  # m^-3
    channel_radius = 40e-6  # m
    ramp_length = 60e-6  # m
    z, y, x = np.meshgrid(info.z, info.y, info.x, indexing="ij")
    density_th = (
        on_axis_density
        * (1 + (x**2 + y**2) / channel_radius**2)
        * np.where(z < ramp_length, z / ramp_length, 1)
    )
    density_th[z < 0] = 0

    # Do not check the 3 first cells and 3 last cells,
    # as this is affected by the edges of the domain
    assert np.all(
        abs(density_th[3:-3, 3:-3, 3:-3] - density_sim[3:-3, 3:-3, 3:-3])
        / abs(density_th[3:-3, 3:-3, 3:-3]).max()
        < 0.02
    )
