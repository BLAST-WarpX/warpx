#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Andrew Myers
# License: BSD-3-Clause-LBNL
#
# This script analyzes the phase-space plot from the 1D particle absorbing boundary test, ensuring that there are not too many fast-moving particles with negative velocities near the boundary

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

ts = OpenPMDTimeSeries("./diags/reducedfiles/PhaseSpaceElectrons")
it = ts.iterations
data, info = ts.get_field(field="data", iteration=8000, plot=False)

# We check the total weight of particles in the region of phase space with z
# between 0 and 50 microns and uz between -5 and -1. If you change the bounds
# or number of points in the PhaseSpaceElectrons diagnostic in the inputs file,
# you need to update the calculations below.
nz = 1000
zmin = -100
zmax = 50
nuz = 1000
uzmin = -20
uzmax = 40

reg_lo_z = 0
reg_hi_z = 50
reg_lo_uz = -5
reg_hi_uz = -1

ilo = int(np.ceil((reg_lo_uz - uzmin) / (uzmax - uzmin) * nuz))
ihi = int(np.ceil((reg_hi_uz - uzmin) / (uzmax - uzmin) * nuz))
jlo = int(np.ceil((reg_lo_z - zmin) / (zmax - zmin) * nz))
jhi = int(np.ceil((reg_hi_z - zmin) / (zmax - zmin) * nz))

# Without the thermalizer the total weight of particles in this region is > 1e22.
assert data[ilo:ihi, jlo:jhi].sum() < 3.2e20
