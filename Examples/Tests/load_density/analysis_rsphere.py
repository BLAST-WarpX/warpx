#!/usr/bin/env python3

# Test loading the plasma density from an openPMD file.
# The file contains a spherically symmetric parabolic density profile:
# n(r) = n0 * (1 + r**2 / channel_radius**2).

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import e

ts = OpenPMDTimeSeries("./diags/diag")

# Loop over all iterations and load the density
for iteration in ts.iterations:
    rho, info = ts.get_field("rho", iteration=iteration)
    density_sim = -rho / e

    # Compute expected density
    on_axis_density = 1e24  # m^-3
    channel_radius = 40e-6  # m

    r = info.x
    density_th = on_axis_density * (1 + r**2 / channel_radius**2)

    # Do not check the 3 first cells and 3 last cells,
    # as this is affected by the edges of the domain
    assert np.all(
        abs(density_th[3:-3] - density_sim[3:-3]) / abs(density_th[3:-3]).max() < 0.03
    )
