#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Ledger tests for radiation-diffusion reflecting and open boundaries."""

import argparse

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["cavity", "reflecting", "vacuum", "marshak"],
    required=True,
)
parser.add_argument("--momentum-component", type=int, choices=[0, 1, 2], default=0)
add_precision_arguments(parser)
args = parser.parse_args()
_, _, cross_dtype = precision_dtypes(args)

rtol = 2.0e-6 if cross_dtype == np.float32 else 2.0e-12

radiation = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
momentum = np.atleast_2d(np.loadtxt("diags/radiation_momentum.txt"))
assert radiation.shape[0] >= 3, "need the initial row plus at least two evolved steps"
assert radiation.shape[1] == 17
assert momentum.shape[0] == radiation.shape[0]
assert momentum.shape[1] == 20

initial = radiation[0, 2]
final = radiation[-1, 2]
cumulative_material = radiation[-1, 6]
cumulative_boundary = radiation[-1, 8]
evolved_total = radiation[1:, 2]
evolved_boundary = radiation[1:, 8]
cumulative_boundary_momentum = momentum[-1, 11:14]
momentum_factor = (
    1.0 if args.mode == "vacuum" else 2.0 / 3.0 if args.mode == "marshak" else 0.0
)
current_momentum_column = 8 + args.momentum_component
cumulative_momentum_column = 11 + args.momentum_component
momentum_atol = 1.0e-18 * max(initial, 1.0) / 299792458.0

print(f"mode:                     {args.mode}")
print(f"initial radiation:        {initial:.16e} J")
print(f"final radiation:          {final:.16e} J")
print(f"cumulative material:      {cumulative_material:.16e} J")
print(f"cumulative boundary loss: {cumulative_boundary:.16e} J")

assert initial > 0.0
np.testing.assert_allclose(radiation[0, 7:9], 0.0, atol=1.0e-30)
np.testing.assert_allclose(momentum[:, 2:8], 0.0, atol=1.0e-24)
np.testing.assert_allclose(radiation[:, 11:13], 0.0, atol=1.0e-30)
np.testing.assert_allclose(radiation[:, 13], radiation[:, 7], rtol=rtol)
np.testing.assert_allclose(radiation[:, 14], radiation[:, 8], rtol=rtol)
np.testing.assert_allclose(radiation[:, 15:17], 0.0, atol=1.0e-30)
np.testing.assert_allclose(momentum[:, 14:20], 0.0, atol=1.0e-24)
np.testing.assert_allclose(
    momentum[:, current_momentum_column],
    momentum_factor * radiation[:, 7] / 299792458.0,
    rtol=rtol,
    atol=momentum_atol,
)
np.testing.assert_allclose(
    momentum[:, cumulative_momentum_column],
    momentum_factor * radiation[:, 8] / 299792458.0,
    rtol=rtol,
    atol=momentum_atol,
)
unused_boundary_columns = [
    column
    for column in range(8, 14)
    if column not in (current_momentum_column, cumulative_momentum_column)
]
np.testing.assert_allclose(
    momentum[:, unused_boundary_columns], 0.0, atol=momentum_atol
)

# final + sources_to_material + escaped = initial
np.testing.assert_allclose(
    final + cumulative_material + cumulative_boundary,
    initial,
    rtol=rtol,
    atol=1.0e-18 * max(initial, 1.0),
)

if args.mode in ("cavity", "reflecting"):
    np.testing.assert_allclose(
        cumulative_boundary, 0.0, atol=1.0e-18 * max(initial, 1.0)
    )
    np.testing.assert_allclose(
        evolved_total,
        evolved_total[0],
        rtol=rtol,
        atol=1.0e-18 * max(initial, 1.0),
    )
    if args.mode == "cavity":
        assert np.max(radiation[:, 3]) > 0.0, "cavity radiation never became streaming"
    np.testing.assert_allclose(
        cumulative_boundary_momentum,
        0.0,
        atol=1.0e-18 * max(initial, 1.0) / 299792458.0,
    )
else:
    assert np.all(np.diff(evolved_total) <= rtol * max(initial, 1.0))
    assert np.all(np.diff(evolved_boundary) >= -1.0e-18 * max(initial, 1.0))
    minimum_loss_fraction = 0.05 if args.mode == "vacuum" else 0.02
    assert cumulative_boundary > minimum_loss_fraction * initial
    assert final < evolved_total[0]
    np.testing.assert_allclose(
        cumulative_boundary_momentum[args.momentum_component],
        momentum_factor * cumulative_boundary / 299792458.0,
        rtol=rtol,
        atol=1.0e-18 * initial / 299792458.0,
    )
    np.testing.assert_allclose(
        np.delete(cumulative_boundary_momentum, args.momentum_component),
        0.0,
        atol=1.0e-18 * initial / 299792458.0,
    )
