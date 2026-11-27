#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--precision", choices=["SINGLE", "DOUBLE"], required=True)
parser.add_argument("--particle-precision", choices=["SINGLE", "DOUBLE"], required=True)
args = parser.parse_args()


def load_reduced(path: Path):
    with path.open() as stream:
        labels = stream.readline().split()
    return labels, np.atleast_2d(np.loadtxt(path))


energy_labels, particle_energy = load_reduced(Path("diags/particle_energy.txt"))
momentum_labels, particle_momentum = load_reduced(Path("diags/particle_momentum.txt"))
_, radiation_energy = load_reduced(Path("diags/radiation_energy.txt"))

electron_energy_column = next(
    index for index, label in enumerate(energy_labels) if label.endswith("electrons(J)")
)
electron_momentum_columns = [
    next(
        index
        for index, label in enumerate(momentum_labels)
        if label.endswith(f"electrons_{direction}(kg*m/s)")
    )
    for direction in "xyz"
]

electron_energy_change = (
    particle_energy[-1, electron_energy_column]
    - particle_energy[0, electron_energy_column]
)
material_energy_gain = radiation_energy[-1, 6]
initial_momentum = particle_momentum[0, electron_momentum_columns]
final_momentum = particle_momentum[-1, electron_momentum_columns]

print(f"electron energy change: {electron_energy_change:.17e} J")
print(f"material energy ledger: {material_energy_gain:.17e} J")
print(f"initial electron momentum: {initial_momentum}")
print(f"final electron momentum:   {final_momentum}")

assert electron_energy_change > 0.0
precision_epsilon = (
    np.finfo(np.float32).eps
    if args.precision == "SINGLE" or args.particle_precision == "SINGLE"
    else np.finfo(np.float64).eps
)
energy_rtol = max(2.0e-10, 8.0 * precision_epsilon)
np.testing.assert_allclose(
    electron_energy_change, material_energy_gain, rtol=energy_rtol, atol=0.0
)
if args.precision == "SINGLE" or args.particle_precision == "SINGLE":
    momentum_scale = np.maximum(np.abs(initial_momentum), np.abs(final_momentum))
    momentum_atol = 8.0 * precision_epsilon * momentum_scale
    momentum_error = np.abs(final_momentum - initial_momentum)
    assert np.all(momentum_error <= momentum_atol), (
        momentum_error,
        momentum_atol,
    )
else:
    np.testing.assert_allclose(
        final_momentum, initial_momentum, rtol=2.0e-12, atol=1.0e-30
    )
