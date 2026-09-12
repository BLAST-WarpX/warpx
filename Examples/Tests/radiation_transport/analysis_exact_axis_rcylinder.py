#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check exact streaming through the axis and into a second radial cell."""

import argparse
from pathlib import Path

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from read_raw_data import _read_buffer

parser = argparse.ArgumentParser()
add_precision_arguments(parser)
args = parser.parse_args()
_, _, cross_dtype = precision_dtypes(args)

plotfile = Path("diags/diag000002")
with open(plotfile / "Header") as header:
    header.readline()
    n_fields = int(header.readline())
    field_names = [header.readline().strip() for _ in range(n_fields)]
fields = _read_buffer(str(plotfile), str(plotfile / "Level_0" / "Cell_H"), field_names)
deposition = np.asarray(fields["radiation_material_energy"]).reshape(-1)

particle_energy = np.atleast_2d(np.loadtxt("diags/particle_energy.txt"))
radiation_energy = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
initial_energy = particle_energy[0, 2]
expected_step_two = initial_energy * np.array(
    [
        np.exp(-2.0 * 0.25) * (1.0 - np.exp(-2.0 * 0.20)),
        np.exp(-2.0 * 0.45) * (1.0 - np.exp(-8.0 * 0.05)),
        0.0,
        0.0,
    ]
)
expected_final = initial_energy * np.exp(-(2.0 * 0.45 + 8.0 * 0.05))
expected_cumulative_deposition = initial_energy - expected_final
rtol = 3.0e-5 if cross_dtype == np.float32 else 8.0e-13
atol = 2.0e-12 * initial_energy

print(f"inner-cell material deposition: {deposition[0]:.16e} J")
print(f"outer-cell material deposition: {deposition[1]:.16e} J")
print(f"final photon energy:             {particle_energy[-1, 2]:.16e} J")

np.testing.assert_array_equal(np.flatnonzero(deposition), np.array([0, 1]))
np.testing.assert_allclose(deposition, expected_step_two, rtol=rtol, atol=atol)
np.testing.assert_allclose(particle_energy[-1, 2], expected_final, rtol=rtol)
np.testing.assert_allclose(radiation_energy[-1, 5], expected_step_two.sum(), rtol=rtol)
np.testing.assert_allclose(
    radiation_energy[-1, 6], expected_cumulative_deposition, rtol=rtol
)
np.testing.assert_allclose(
    particle_energy[-1, 2] + radiation_energy[-1, 6],
    initial_energy,
    rtol=rtol,
    atol=atol,
)
