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

ts = OpenPMDTimeSeries("diags/reducedfiles/PhaseSpaceElectrons")
it = ts.iterations
data, info = ts.get_field(field="data", iteration=8000, plot=False)
