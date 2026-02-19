#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Andrew Myers
# License: BSD-3-Clause-LBNL
#
# This script plots the phase space diagram using the reduced diagnostics from the particle absorbing boundary test case.

import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

ts = OpenPMDTimeSeries("diags/reducedfiles/PhaseSpaceElectrons")
it = ts.iterations
data, info = ts.get_field(field="data", iteration=8000, plot=True)
plt.pcolormesh(np.log10(data))
ax = plt.gca()
ax.set_yticks([0, 333.33333, 666.666667, 1000])
ax.set_yticklabels([-20, 0, 20, 40])
ax.set_xticks([0, 200, 400, 600, 800, 1000])
ax.set_xticklabels([-100, -50, 0, 50, 100, 150])
ax.set_ylabel(r"$uz [m c]$")
plt.savefig("thermalizer")
