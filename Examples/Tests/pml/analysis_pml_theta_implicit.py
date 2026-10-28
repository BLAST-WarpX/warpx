#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL

"""Check absorption of a vacuum pulse, as in the explicit PML tests."""

import numpy as np
from scipy.constants import epsilon_0

data = np.loadtxt("diags/reducedfiles/field_energy.txt")
energy = data[:, 2]
# Both initial Gaussian polarizations have energy epsilon_0*pi*width^2/2.
# The diagnostic integrates energy over the physical domain, excluding the PML.
initial_energy = epsilon_0 * np.pi * (2.0e-6) ** 2
remaining = energy[-1] / initial_energy
print(f"Remaining energy fraction: {remaining}")
assert np.isfinite(data).all()
assert np.max(energy) < 1.01 * initial_energy
assert remaining < 0.01
