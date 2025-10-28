#!/usr/bin/env python3

# Copyright 2024, Remi Lehe
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This script tests the conservation of energy for a thermal plasma with periodic boundary.
# Here, we use energy-converving gather and an electrostatic solver. The energy
# is not expected to be exactly conserved, but it is expected to be better conserved
# than other gathering scheme. This tests checks that the energy does not increase by
# more than 0.3% over the duration of the simulatoin.

import numpy as np

# Get energy as a function of time, from reduced diagnostics
EFdata = np.genfromtxt("./diags/reducedfiles/EF.txt")  # Field energy
EPdata = np.genfromtxt("./diags/reducedfiles/EP.txt")  # Particle energy
field_energy = EFdata[:, 2]
particle_energy = EPdata[:, 2]
E = field_energy + particle_energy
print(abs(E - E[0]) / E[0])
# Check that the energy is conserved to 0.3%
assert np.all(abs(E - E[0]) / E[0] < 0.003)
