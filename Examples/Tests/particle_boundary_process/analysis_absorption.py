#!/usr/bin/env python3

# This test shoots a beam of electrons at cubic embedded boundary geometry
# At time step 40, none of the particles have hit the boundary yet. At time
# step 60, all of them should have been absorbed by the boundary. In the
# absence of the cube, none of the particles would have had time to exit
# the problem domain yet.

import sys

import yt

# all particles are still there
ds40 = yt.load("diags/diag1000040")
np40 = ds40.index.particle_headers["electrons"].num_particles
assert np40 == 612

# all particles have been removed
filename = sys.argv[1]
ds60 = yt.load(filename)
np60 = ds60.index.particle_headers["electrons"].num_particles
assert np60 == 0
