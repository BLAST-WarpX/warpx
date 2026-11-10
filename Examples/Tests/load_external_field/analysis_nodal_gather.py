#!/usr/bin/env python3

"""An affine, curl-free field must survive coarse/fine nodal gathering."""

import numpy as np

samples = np.loadtxt("diags/reducedfiles/probe.txt", skiprows=1, ndmin=2)
np.testing.assert_array_equal(samples[:, 0], [0, 1])
x, z = samples[:, 2], samples[:, 4]
expected = np.column_stack((1 + x, np.full_like(x, 3), 2 + z))
np.testing.assert_allclose(samples[:, 5:8], expected, rtol=2e-6, atol=0)
