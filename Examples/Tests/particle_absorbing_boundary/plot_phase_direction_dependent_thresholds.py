#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Andrew Myers, Luca Fedeli
# License: BSD-3-Clause-LBNL
#
# This script plots the phase space diagram using the reduced diagnostics from the
# 1D particle absorbing boundary with direction dependent thresholds test case.

import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

data = {}

filenames = [
    "PhaseSpaceElectrons_zux",
    "PhaseSpaceElectrons_zuy",
    "PhaseSpaceElectrons",
]
keys = ["zux", "zuy", "zuz"]
yticks = [(-2, -1, 1, 2), (-2, -1, 1, 2), (-20, 0, 20, 40)]

for filename, key in zip(filenames, keys):
    print(filename, key)
    ts = OpenPMDTimeSeries("diags/reducedfiles/" + filename)
    data[key], _ = ts.get_field(field="data", iteration=8000, plot=True)

fig, ax = plt.subplots(3, 1)

for i, key in enumerate(keys):
    ax[i].pcolormesh(np.log10(data[key]))
    ax[i].set_yticks([0, 333.33333, 666.666667, 1000])
    ax[i].set_yticklabels(yticks[i])
    ax[i].set_xticks([0, 333.33333, 666.66667, 1000])
    ax[i].set_xticklabels([-100, -50, 0, 50])
    ax[i].set_xlabel(r"$z (\mu m)$")
    ax[i].set_ylabel(key[1:] + r"$ [m c]$")

plt.tight_layout()
plt.savefig("thermalizer_direction_dependent_thresholds.png", dpi=200)
