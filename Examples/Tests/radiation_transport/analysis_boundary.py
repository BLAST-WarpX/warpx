#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--precision", choices=["SINGLE", "DOUBLE"], required=True)
parser.add_argument("--particle-precision", choices=["SINGLE", "DOUBLE"], required=True)
parser.add_argument("--mode", choices=["serial", "mpi"], required=True)
args = parser.parse_args()

single_precision = args.precision == "SINGLE" or args.particle_precision == "SINGLE"
rtol = 2.0e-6 if single_precision else 3.0e-13

radiation_energy = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
radiation_momentum = np.atleast_2d(np.loadtxt("diags/radiation_momentum.txt"))
initial_radiation = radiation_energy[0, 2]

print(f"initial radiation:        {initial_radiation:.16e} J")
print(f"final radiation:          {radiation_energy[-1, 2]:.16e} J")
print(f"boundary energy loss:     {radiation_energy[-1, 7]:.16e} J")
print(f"cumulative boundary loss: {radiation_energy[-1, 8]:.16e} J")

assert initial_radiation > 0.0
assert radiation_energy.shape[1] == 17
assert radiation_momentum.shape == (radiation_energy.shape[0], 20)
np.testing.assert_allclose(radiation_energy[-1, 2:7], 0.0, atol=1.0e-30)
np.testing.assert_allclose(radiation_energy[-1, 7], initial_radiation, rtol=rtol)
np.testing.assert_allclose(radiation_energy[-1, 8], initial_radiation, rtol=rtol)
np.testing.assert_allclose(radiation_energy[-1, 9:11], 0.0, atol=1.0e-30)
np.testing.assert_allclose(radiation_energy[-1, 11], initial_radiation, rtol=rtol)
np.testing.assert_allclose(radiation_energy[-1, 12], initial_radiation, rtol=rtol)
np.testing.assert_allclose(radiation_energy[:, 13:17], 0.0, atol=1.0e-30)
np.testing.assert_allclose(
    radiation_energy[:, 7],
    radiation_energy[:, 11] + radiation_energy[:, 13],
    rtol=rtol,
    atol=1.0e-30,
)
np.testing.assert_allclose(
    radiation_energy[:, 8],
    radiation_energy[:, 12] + radiation_energy[:, 14],
    rtol=rtol,
    atol=1.0e-30,
)

if args.mode == "serial":
    # The packet exits the high-z face, so its signed escaped momentum is +E/c.
    np.testing.assert_allclose(radiation_momentum[:, 2:14], 0.0, atol=1.0e-30)
    np.testing.assert_allclose(
        radiation_momentum[:, 16],
        radiation_energy[:, 11] / 299792458.0,
        rtol=rtol,
        atol=1.0e-30,
    )
    np.testing.assert_allclose(
        radiation_momentum[:, 19],
        radiation_energy[:, 12] / 299792458.0,
        rtol=rtol,
        atol=1.0e-30,
    )
    np.testing.assert_allclose(
        radiation_momentum[:, [14, 15, 17, 18]], 0.0, atol=1.0e-30
    )
else:
    # Packets with low:high weights 1:2 escape through opposite z faces.  Total
    # escaped energy and net +z momentum E/(3c) independently constrain both
    # face contributions while every other boundary component remains zero.
    np.testing.assert_allclose(radiation_momentum[:, 2:14], 0.0, atol=1.0e-30)
    np.testing.assert_allclose(
        radiation_momentum[:, [14, 15, 17, 18]], 0.0, atol=1.0e-30
    )
    np.testing.assert_allclose(
        radiation_momentum[:, 16],
        radiation_energy[:, 11] / (3.0 * 299792458.0),
        rtol=rtol,
        atol=1.0e-30,
    )
    np.testing.assert_allclose(
        radiation_momentum[:, 19],
        radiation_energy[:, 12] / (3.0 * 299792458.0),
        rtol=rtol,
        atol=1.0e-30,
    )
