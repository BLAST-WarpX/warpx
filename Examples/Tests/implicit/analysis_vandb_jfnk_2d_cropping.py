#!/usr/bin/env python3

# Copyright 2024 Justin Angus
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from the script `inputs_vandb_2d`.
# This simulates a 2D periodic plasma using the implicit solver
# with the Villasenor deposition using shape factor 2.
import sys

import numpy as np
import yt
from scipy.constants import e, epsilon_0

# check for machine precision conservation of charge density
n0 = 1.0e12

pltdir = sys.argv[1]
ds = yt.load(pltdir)
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

divE = data["boxlib", "divE"].value
rho = data["boxlib", "rho"].value

# compute local error in Gauss's law
drho = (rho - epsilon_0 * divE) / e / n0

# compute RMS on in error on the grid
nX = drho.shape[0]
nZ = drho.shape[1]
drho_max = np.abs(drho).max()

tolerance_max_charge = 1.0e-13

print(f"max error in charge conservation: {drho_max}")
print(f"tolerance: {tolerance_max_charge}")

assert drho_max < tolerance_max_charge
