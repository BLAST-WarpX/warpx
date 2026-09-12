#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Check every radial-loading quadrature site, not just the total mass."""

import argparse

import numpy as np
import yt

parser = argparse.ArgumentParser()
parser.add_argument("--power", type=int, required=True)
parser.add_argument("--precision", choices=["SINGLE", "DOUBLE"], required=True)
args = parser.parse_args()
yt.set_log_level(50)
ds = yt.load("diags/plt000000")
data = ds.all_data()
radius = data["electrons", "particle_position_x"].v
z = data["electrons", "particle_position_y"].v
weight = data["electrons", "particle_weight"].v

# Independent deterministic quadrature: 4 sites per cell per active direction.
# Cylindrical loading maps a uniform logical coordinate xi to r=xi^(1/(p+1)).
logical = (np.arange(128) + 0.5) / 128
radial_sites = logical ** (1 / (args.power + 1))
axial_sites = -1 + 2 * (np.arange(256) + 0.5) / 256
rr, zz = np.meshgrid(radial_sites, axial_sites, indexing="ij")
mask = (rr * rr + zz * zz > 0.25**2) & (rr * rr + zz * zz < 0.65**2)
expected_r, expected_z = rr[mask], zz[mask]
expected_weight = (
    1e20
    * 2
    * np.pi
    / (args.power + 1)
    * expected_r ** (1 - args.power)
    / 128
    * (2 / 256)
)
print(f"power={args.power}; actual particles={radius.size}; expected={mask.sum()}")
assert radius.size == mask.sum(), (
    "Logical-coordinate culling removed physical profile samples"
)

# Map output positions to their unique quadrature indices. This also catches
# duplicate samples or a compensating weight change that hides missing particles.
ir = np.argmin(abs(radius[:, None] - radial_sites[None, :]), axis=1)
iz = np.argmin(abs(z[:, None] - axial_sites[None, :]), axis=1)
order = np.argsort(ir * 256 + iz)
assert np.array_equal(np.sort(ir * 256 + iz), np.flatnonzero(mask))
eps = np.finfo(np.float32 if args.precision == "SINGLE" else np.float64).eps
np.testing.assert_allclose(radius[order], expected_r, rtol=64 * eps, atol=64 * eps)
np.testing.assert_allclose(z[order], expected_z, rtol=64 * eps, atol=64 * eps)
np.testing.assert_allclose(weight[order], expected_weight, rtol=64 * eps, atol=0)
assert np.count_nonzero(mask & (rr < 0.4)) > 0
assert np.all(weight > 0)
print("All physical sites and cylindrical weights match the independent quadrature.")
