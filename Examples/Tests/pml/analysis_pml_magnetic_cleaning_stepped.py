#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Require bounded vacuum fields at stepped refinement/PML interfaces."""

import numpy as np

energy = np.loadtxt("diags/reducedfiles/field_energy.txt")
number = np.loadtxt("diags/reducedfiles/particle_number.txt")
assert energy.shape[1] == 8, "Expect parent and fine-patch energy columns"
assert energy[0, 0] == 0 and energy[-1, 0] == 250
assert np.isfinite(energy).all() and np.isfinite(number).all()
assert np.all(number[:, 2] == 0), "This test must remain particle-free"

# Check each patch independently: summing overlapping levels would not give
# composite AMR energy. Allow leapfrog/cleaning transients, but reject gain from
# the unforced zero-particle system. With magnetic cleaning disabled, the fine
# energy grows by more than 1e9 over this same 250-step window.
patch_energy = energy[:, (2, 5)]
initial = patch_energy[0]
assert np.all(initial > 0)
assert np.all(patch_energy <= 2 * initial), patch_energy.max(axis=0) / initial
assert patch_energy[-1, 1] < initial[1], "The fine-grid wave should lose energy to PML"
