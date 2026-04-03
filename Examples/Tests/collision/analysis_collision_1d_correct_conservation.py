#!/usr/bin/env python3

# Copyright 2025 David Grote
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from the script `inputs_test_1d_collision_z_correct_conservation`.
# run locally: python analysis_collision_1d_correct_conservation.py diags/diag1000000 diags/diag1000010
#
# This is a 1D inter- and intra-species Coulomb scattering test consisting
# of a nonuniform low-density population streaming into a higher nonuniform density population at rest.
# Both populations are the same carbon12 ion.
# This checks the conservation of energy, which is enforced by applying a correction after the collisions.
#
import sys

import numpy as np
import yt
from scipy import constants

# this will be the name of the plot file
first_fn = sys.argv[1]
last_fn = sys.argv[2]

# tolerance for the relative change in the momentum and energy
# The tolerance on the energy is larger since the calculation of
# the correction can have large numerical errors due to its nature,
# involving the difference between similar numbers.
p_tolerance = 1.0e-14
KE_tolerance = 1.0e-10

ds0 = yt.load(first_fn)
ds1 = yt.load(last_fn)

ad0 = ds0.all_data()
ad1 = ds1.all_data()

# carbon 12 ion (mass = 12*amu - 6*me)
m_c12 = 12.0 * constants.m_u - 6.0 * constants.m_e
mA = m_c12
mB = m_c12

wwA0 = ad0["ionsA", "particle_weight"].value
wwB0 = ad0["ionsB", "particle_weight"].value

pxA0 = ad0["ionsA", "particle_momentum_x"].value * wwA0
pyA0 = ad0["ionsA", "particle_momentum_y"].value * wwA0
pzA0 = ad0["ionsA", "particle_momentum_z"].value * wwA0

pxB0 = ad0["ionsB", "particle_momentum_x"].value * wwB0
pyB0 = ad0["ionsB", "particle_momentum_y"].value * wwB0
pzB0 = ad0["ionsB", "particle_momentum_z"].value * wwB0

wwA1 = ad1["ionsA", "particle_weight"].value
wwB1 = ad1["ionsB", "particle_weight"].value

pxA1 = ad1["ionsA", "particle_momentum_x"].value * wwA1
pyA1 = ad1["ionsA", "particle_momentum_y"].value * wwA1
pzA1 = ad1["ionsA", "particle_momentum_z"].value * wwA1

pxB1 = ad1["ionsB", "particle_momentum_x"].value * wwB1
pyB1 = ad1["ionsB", "particle_momentum_y"].value * wwB1
pzB1 = ad1["ionsB", "particle_momentum_z"].value * wwB1

pxA0sum = pxA0.sum()
pyA0sum = pyA0.sum()
pzA0sum = pzA0.sum()

pxB0sum = pxB0.sum()
pyB0sum = pyB0.sum()
pzB0sum = pzB0.sum()

pxA1sum = pxA1.sum()
pyA1sum = pyA1.sum()
pzA1sum = pzA1.sum()

pxB1sum = pxB1.sum()
pyB1sum = pyB1.sum()
pzB1sum = pzB1.sum()

print("step 0")
print(f"A momenta {pxA0sum} {pyA0sum} {pzA0sum}")
print(f"B momenta {pxB0sum} {pyB0sum} {pzB0sum}")
print("step 1")
print(f"A momenta {pxA1sum} {pyA1sum} {pzA1sum}")
print(f"B momenta {pxB1sum} {pyB1sum} {pzB1sum}")
print(
    f"totals 0  {pxA0sum + pxB0sum:24.16e} {pyA0sum + pyB0sum:24.16e} {pzA0sum + pzB0sum:24.16e}"
)
print(
    f"totals 1  {pxA1sum + pxB1sum:24.16e} {pyA1sum + pyB1sum:24.16e} {pzA1sum + pzB1sum:24.16e}"
)

px_rel = (pxA1sum + pxB1sum - pxA0sum - pxB0sum) / (pxA0sum + pxB0sum)
py_rel = (pyA1sum + pyB1sum - pyA0sum - pyB0sum) / (pyA0sum + pyB0sum)
pz_rel = (pzA1sum + pzB1sum - pzA0sum - pzB0sum) / (pzA0sum + pzB0sum)

print(f"relative momentum change {px_rel} {py_rel} {pz_rel}, tolerance {p_tolerance}")


def KE(ad, ss, mass):
    ux = ad[ss, "particle_momentum_x"].value / mass
    uy = ad[ss, "particle_momentum_y"].value / mass
    uz = ad[ss, "particle_momentum_z"].value / mass
    u2 = ux * ux + uy * uy + uz * uz
    gamma = np.sqrt(1.0 + u2 / constants.c**2)
    return 1.0 / (1.0 + gamma) * mass * u2


KEA0 = KE(ad0, "ionsA", mA) * wwA0
KEB0 = KE(ad0, "ionsB", mB) * wwB0
KEA1 = KE(ad1, "ionsA", mA) * wwA1
KEB1 = KE(ad1, "ionsB", mB) * wwB1

KEA0sum = KEA0.sum()
KEB0sum = KEB0.sum()

KEA1sum = KEA1.sum()
KEB1sum = KEB1.sum()

print()
print("step 0")
print(f"A KE {KEA0sum}")
print(f"B KE {KEB0sum}")
print("step 1")
print(f"A KE {KEA1sum}")
print(f"B KE {KEB1sum}")
print(f"totals 0  {KEA0sum + KEB0sum:24.16e}")
print(f"totals 1  {KEA1sum + KEB1sum:24.16e}")

KE_rel = (KEA1sum + KEB1sum - KEA0sum - KEB0sum) / (KEA0sum + KEB0sum)

print(f"relative energy change {KE_rel}, tolerance {KE_tolerance}")

assert np.abs(px_rel) < p_tolerance
assert np.abs(py_rel) < p_tolerance
assert np.abs(pz_rel) < p_tolerance
assert np.abs(KE_rel) < KE_tolerance
