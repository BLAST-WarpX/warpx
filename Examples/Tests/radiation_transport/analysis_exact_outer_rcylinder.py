#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check exact attenuation to an open RCYLINDER outer face and its ledger."""

import argparse

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes

parser = argparse.ArgumentParser()
add_precision_arguments(parser)
args = parser.parse_args()
_, _, cross_dtype = precision_dtypes(args)

particle_energy = np.atleast_2d(np.loadtxt("diags/particle_energy.txt"))
radiation_energy = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
initial_energy = particle_energy[0, 2]
light_speed = np.float32(299792458.0) if args.precision == "SINGLE" else 299792458.0
path_length = float(light_speed) / 2.0**32
expected_material = initial_energy * (1.0 - np.exp(-2.0 * path_length))
expected_boundary = initial_energy * np.exp(-2.0 * path_length)
rtol = 3.0e-5 if cross_dtype == np.float32 else 8.0e-13
atol = 2.0e-12 * initial_energy

print(f"material energy gain:            {radiation_energy[-1, 5]:.16e} J")
print(f"streaming boundary loss:         {radiation_energy[-1, 11]:.16e} J")
print(f"remaining radiation:             {radiation_energy[-1, 2]:.16e} J")

np.testing.assert_allclose(radiation_energy[-1, 2], 0.0, rtol=0.0, atol=atol)
np.testing.assert_allclose(particle_energy[-1, 2], 0.0, rtol=0.0, atol=atol)
np.testing.assert_allclose(radiation_energy[-1, 5], expected_material, rtol=rtol)
np.testing.assert_allclose(radiation_energy[-1, 11], expected_boundary, rtol=rtol)
np.testing.assert_allclose(
    radiation_energy[-1, 5] + radiation_energy[-1, 11],
    initial_energy,
    rtol=rtol,
    atol=atol,
)
np.testing.assert_allclose(
    radiation_energy[-1, 7], radiation_energy[-1, 11], rtol=rtol, atol=atol
)
