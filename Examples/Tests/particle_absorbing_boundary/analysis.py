#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Andrew Myers
# License: BSD-3-Clause-LBNL
#
# This script analyzes the phase-space plot from the 1D particle absorbing boundary test, ensuring that there are not too many fast-moving particles with negative velocities near the boundary

import sys

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

sys.path.append("../../../Tools/Parser/")
from input_file_parser import parse_input_file

ts = OpenPMDTimeSeries("./diags/reducedfiles/PhaseSpaceElectrons")
it = ts.iterations
data, info = ts.get_field(field="data", iteration=8000, plot=False)

# We check the total weight of particles in the region of phase space with z
# between 0 and 50 microns and uz between -5 and -1.
input_dict = parse_input_file("./warpx_used_inputs")
nz = int(input_dict["PhaseSpaceElectrons.bin_number_abs"][0])
zmin = float(input_dict["PhaseSpaceElectrons.bin_min_abs"][0])
zmax = float(input_dict["PhaseSpaceElectrons.bin_max_abs"][0])
nuz = int(input_dict["PhaseSpaceElectrons.bin_number_ord"][0])
uzmin = float(input_dict["PhaseSpaceElectrons.bin_min_ord"][0])
uzmax = float(input_dict["PhaseSpaceElectrons.bin_max_ord"][0])

reg_lo_z = 0.0
reg_hi_z = 50.0e-6
reg_lo_uz = -5
reg_hi_uz = -1

ilo = int(np.ceil((reg_lo_uz - uzmin) / (uzmax - uzmin) * nuz))
ihi = int(np.ceil((reg_hi_uz - uzmin) / (uzmax - uzmin) * nuz))
jlo = int(np.ceil((reg_lo_z - zmin) / (zmax - zmin) * nz))
jhi = int(np.ceil((reg_hi_z - zmin) / (zmax - zmin) * nz))

# Without the thermalizer the total weight of particles in this region is > 1e22.
assert data[ilo:ihi, jlo:jhi].sum() < 3.2e20
