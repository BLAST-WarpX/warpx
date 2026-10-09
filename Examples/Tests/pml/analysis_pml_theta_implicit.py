#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL

"""Check absorption of a vacuum pulse, as in the explicit PML tests."""

import numpy as np

data = np.loadtxt("diags/reducedfiles/field_energy.txt")
energy = data[:, 2]
# The diagnostic includes the initial state and excludes the PML region.
assert data[0, 1] == 0.0
initial_energy = energy[0]
assert initial_energy > 0.0
remaining = energy[-1] / initial_energy
print(f"Remaining energy fraction: {remaining}")
assert np.isfinite(data).all()
assert np.max(energy) < 1.01 * initial_energy
assert remaining < 0.01
