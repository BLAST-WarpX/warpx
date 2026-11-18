#!/usr/bin/env python3
#
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Peter Scherpelz (Helion Energy)
#
# License: BSD-3-Clause-LBNL

## Velocity coincidence thinning conserves the total weight. With reflecting
## boundaries, no particle leaves the domain, so the weight must stay constant
## while the number of macroparticles decreases.

import numpy as np

# columns: [2] total number of macroparticles, [4] total weight
data = np.loadtxt("diags/reducedfiles/particle_number.txt")
num_macroparticles = data[:, 2]
weight = data[:, 4]
print("macroparticles:", num_macroparticles[0], "->", num_macroparticles[-1])
print("max relative weight change:", np.max(np.abs(weight / weight[0] - 1.0)))

assert num_macroparticles[-1] < num_macroparticles[0]
assert np.allclose(weight, weight[0], rtol=1e-12, atol=0.0)
