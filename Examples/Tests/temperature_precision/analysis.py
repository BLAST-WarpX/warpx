#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Each cell has four symmetric samples with mean zero and variance u0**2/2."""

import argparse

import numpy as np
import yt

parser = argparse.ArgumentParser()
parser.add_argument("--precision", choices=["SINGLE", "DOUBLE"], required=True)
args = parser.parse_args()
ds = yt.load("diags/plt000000")
temperature = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)[
    "boxlib", "T_electrons"
].v
# Match the CODATA constants used by ablastr/constant.H.
expected = 9.1093837139e-31 / 1.602176634e-19 * (299792458.0 * 0.001) ** 2 / 2
eps = np.finfo(np.float32 if args.precision == "SINGLE" else np.float64).eps
assert np.all(np.isfinite(temperature))
assert np.all(temperature > 0), "A representable temperature must not underflow to zero"
np.testing.assert_allclose(temperature, expected, rtol=512 * eps, atol=0)
print(
    f"Temperature range: {temperature.min()} .. {temperature.max()} eV; expected {expected}"
)
