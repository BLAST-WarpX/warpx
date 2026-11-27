#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check face-exact RCYLINDER attenuation across a circular interface."""

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
deposition = np.asarray(fields["radiation_material_energy"]).reshape(-1)

particle_energy = np.atleast_2d(np.loadtxt("diags/particle_energy.txt"))
radiation_energy = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
initial_energy = particle_energy[0, 2]

expected = initial_energy * np.array(
    [0.0, 0.09516258196404048, 0.29830675832332609, 0.0]
)
expected_final = initial_energy * 0.60653065971263342
rtol = 3.0e-5 if cross_dtype == np.float32 else 8.0e-13
atol = 2.0e-12 * initial_energy

print(f"initial photon energy:          {initial_energy:.16e} J")
print(f"cell-1 material deposition:     {deposition[1]:.16e} J")
print(f"cell-2 material deposition:     {deposition[2]:.16e} J")
print(f"analytic final photon energy:   {expected_final:.16e} J")
print(f"simulated final photon energy:  {particle_energy[-1, 2]:.16e} J")

np.testing.assert_array_equal(np.flatnonzero(deposition), np.array([1, 2]))
np.testing.assert_allclose(deposition, expected, rtol=rtol, atol=atol)
np.testing.assert_allclose(particle_energy[-1, 2], expected_final, rtol=rtol)
np.testing.assert_allclose(radiation_energy[-1, 5], np.sum(expected), rtol=rtol)
np.testing.assert_allclose(
    particle_energy[-1, 2] + np.sum(deposition), initial_energy, rtol=rtol, atol=atol
)
np.testing.assert_allclose(radiation_energy[-1, 2], particle_energy[-1, 2], rtol=rtol)
