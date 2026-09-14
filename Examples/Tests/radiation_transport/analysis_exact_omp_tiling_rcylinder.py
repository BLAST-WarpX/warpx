#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check race-free exact RCYLINDER scatter across CPU particle tiles."""

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
packet_energy = initial_energy / 32.0
expected = packet_energy * np.array(
    [
        0.0,
        16.0 * (1.0 - np.exp(-2.0 * 0.05)),
        16.0 * np.exp(-2.0 * 0.05) * (1.0 - np.exp(-8.0 * 0.05))
        + 16.0 * (1.0 - np.exp(-8.0 * 0.1)),
        0.0,
    ]
)
expected_final = packet_energy * (
    16.0 * np.exp(-(2.0 * 0.05 + 8.0 * 0.05)) + 16.0 * np.exp(-8.0 * 0.1)
)
rtol = 4.0e-5 if cross_dtype == np.float32 else 8.0e-13
atol = 3.0e-12 * initial_energy

print(f"tile-crossing cell deposition: {deposition[1]:.16e} J")
print(f"contended-cell deposition:     {deposition[2]:.16e} J")
print(f"final photon energy:           {particle_energy[-1, 2]:.16e} J")

np.testing.assert_array_equal(np.flatnonzero(deposition), np.array([1, 2]))
np.testing.assert_allclose(deposition, expected, rtol=rtol, atol=atol)
np.testing.assert_allclose(particle_energy[-1, 2], expected_final, rtol=rtol)
np.testing.assert_allclose(radiation_energy[-1, 5], expected.sum(), rtol=rtol)
np.testing.assert_allclose(
    particle_energy[-1, 2] + deposition.sum(), initial_energy, rtol=rtol, atol=atol
)
