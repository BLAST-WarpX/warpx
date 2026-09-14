#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""An independent energy/trajectory oracle for selective photon escape."""

import argparse
from pathlib import Path

import numpy as np
import yt

parser = argparse.ArgumentParser()
parser.add_argument("--precision", choices=["SINGLE", "DOUBLE"], required=True)
parser.add_argument("--particle-precision", choices=["SINGLE", "DOUBLE"], required=True)
parser.add_argument("--initial", default="diags/plt000000")
args = parser.parse_args()
# Geometry, timestep and reduced energy use field precision; positions and
# momenta use particle precision. The analytic trajectory depends on both.
eps = max(
    np.finfo(np.float32 if precision == "SINGLE" else np.float64).eps
    for precision in (args.precision, args.particle_precision)
)
yt.set_log_level(50)
initial = yt.load(args.initial).all_data()
final = yt.load("diags/plt000002").all_data()
initial_energy = np.sum(
    initial["photons", "particle_weight"].v
    * 299792458.0
    * np.sqrt(sum(initial["photons", "particle_momentum_" + a].v ** 2 for a in "xyz"))
)
assert final["photons", "particle_weight"].size == 1
np.testing.assert_allclose(
    final["photons", "particle_position_x"].v, 0.08, rtol=128 * eps, atol=128 * eps
)
np.testing.assert_allclose(
    final["photons", "particle_position_y"].v, 0, rtol=0, atol=128 * eps
)
assert final["tracer", "particle_weight"].size == 1
expected_radius = 2 - (0.995 + 0.10 * 0.2 / np.sqrt(1 + 0.2**2))
np.testing.assert_allclose(
    final["tracer", "particle_position_x"].v,
    expected_radius,
    rtol=128 * eps,
    atol=128 * eps,
)
np.testing.assert_allclose(
    final["tracer", "particle_momentum_x"].v,
    -initial["tracer", "particle_momentum_x"].v,
    rtol=128 * eps,
    atol=0,
)
ledger = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
np.testing.assert_allclose(ledger[-1, 2], initial_energy / 3, rtol=128 * eps, atol=0)
np.testing.assert_allclose(
    ledger[-1, 8], 2 * initial_energy / 3, rtol=128 * eps, atol=0
)
assert (
    Path("diags/chk000002/RadiationPhotonBoundary_data.txt").read_text().strip()
    == "absorbing_nonperiodic_v1 photons"
)
print(
    "Radial/axial photon escape, axis transit, material reflection and energy ledger pass."
)
