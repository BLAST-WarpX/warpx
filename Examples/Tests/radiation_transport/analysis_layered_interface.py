#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse
from pathlib import Path

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from read_raw_data import _read_buffer

parser = argparse.ArgumentParser()
add_precision_arguments(parser)
args = parser.parse_args()
_, _, cross_dtype = precision_dtypes(args)

plotfile = Path("diags/diag000001")
with open(plotfile / "Header") as header:
    header.readline()
    n_fields = int(header.readline())
    field_names = [header.readline().strip() for _ in range(n_fields)]

fields = _read_buffer(str(plotfile), str(plotfile / "Level_0" / "Cell_H"), field_names)
deposition = np.squeeze(fields["radiation_material_energy"])
particle_energy = np.atleast_2d(np.loadtxt("diags/particle_energy.txt"))
radiation_energy = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))

initial_energy = particle_energy[0, 2]
final_energy = particle_energy[-1, 2]
alpha_lo = 2.0
alpha_hi = 8.0
distance_lo = 0.05
distance_hi = 0.05

after_lo = initial_energy * np.exp(-alpha_lo * distance_lo)
expected_final = after_lo * np.exp(-alpha_hi * distance_hi)
expected = np.zeros(8)
expected[3] = initial_energy - after_lo
expected[4] = after_lo - expected_final

cross_rtol = 3.0e-5 if cross_dtype == np.float32 else 8.0e-13
atol = 2.0e-12 * initial_energy

print(f"initial photon energy:          {initial_energy:.16e} J")
print(f"low-opacity layer deposition:   {deposition[3]:.16e} J")
print(f"high-opacity layer deposition:  {deposition[4]:.16e} J")
print(f"analytic final photon energy:   {expected_final:.16e} J")
print(f"simulated final photon energy:  {final_energy:.16e} J")

np.testing.assert_allclose(deposition, expected, rtol=cross_rtol, atol=atol)
np.testing.assert_allclose(final_energy, expected_final, rtol=cross_rtol)
np.testing.assert_allclose(radiation_energy[-1, 6], np.sum(expected), rtol=cross_rtol)
np.testing.assert_allclose(
    final_energy + np.sum(deposition), initial_energy, rtol=cross_rtol
)
assert np.flatnonzero(deposition).tolist() == [3, 4]
