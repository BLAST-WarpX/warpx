#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Escaped packet energy and (r,theta,z) momentum in RCYLINDER."""

import numpy as np
from scipy.constants import c

radiation = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
momentum = np.atleast_2d(np.loadtxt("diags/radiation_momentum.txt"))

assert radiation.shape == (2, 17)
assert momentum.shape == (2, 20)

initial_energy = radiation[0, 2]
expected_component = initial_energy / (np.sqrt(2.0) * c)

print(f"escaped streaming energy: {radiation[-1, 11]:.16e} J")
print(f"escaped radial momentum:  {momentum[-1, 14]:.16e} kg*m/s")
print(f"escaped axial momentum:   {momentum[-1, 16]:.16e} kg*m/s")

assert initial_energy > 0.0
np.testing.assert_allclose(radiation[-1, 2:7], 0.0, atol=1.0e-30)
np.testing.assert_allclose(radiation[-1, 7:9], initial_energy, rtol=5.0e-6)
np.testing.assert_allclose(radiation[-1, 9:11], 0.0, atol=1.0e-30)
np.testing.assert_allclose(radiation[-1, 11:13], initial_energy, rtol=5.0e-6)
np.testing.assert_allclose(radiation[:, 13:17], 0.0, atol=1.0e-30)
np.testing.assert_allclose(momentum[:, 2:14], 0.0, atol=1.0e-30)
np.testing.assert_allclose(momentum[-1, [14, 16]], expected_component, rtol=5.0e-6)
np.testing.assert_allclose(momentum[-1, [17, 19]], expected_component, rtol=5.0e-6)
np.testing.assert_allclose(momentum[:, [15, 18]], 0.0, atol=1.0e-20)
