#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Independent Planck quadrature for a nonuniform initialized radiation bath."""

import argparse
from pathlib import Path

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from read_raw_data import _read_buffer
from scipy.integrate import quad

parser = argparse.ArgumentParser(description=__doc__)
add_precision_arguments(parser)
args = parser.parse_args()
field_dtype, _, _ = precision_dtypes(args)
path = Path("diags/diag1000000")
with (path / "Header").open() as header:
    header.readline()
    names = [header.readline().strip() for _ in range(int(header.readline()))]
fields = _read_buffer(str(path), str(path / "Level_0/Cell_H"), names)
temperature = 10000 * (1 + 0.1 * np.cos(2 * np.pi * (np.arange(16) + 0.5) / 16))
fractions = np.array(
    [
        quad(lambda x: x**3 / np.expm1(x), 0, 1e-19 / (1.380649e-23 * t))[0]
        * 15
        / np.pi**4
        for t in temperature
    ]
)
total = 7.565733250280007e-16 * temperature**4 / 16
rtol = 6e-6 if field_dtype == np.float32 else 3e-12
for group, fraction in enumerate((fractions, 1 - fractions)):
    actual = np.asarray(fields[f"radiation_diffusion_energy_g{group}"]).squeeze()
    assert np.all(np.isfinite(actual)) and np.all(actual >= 0)
    np.testing.assert_allclose(actual, total * fraction, rtol=rtol)
print("Nonuniform initial blackbody groups match independent Planck quadrature")
