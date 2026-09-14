#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""A zero-length face departure must not convert a streaming packet."""

import argparse
from pathlib import Path

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from read_raw_data import _read_buffer

parser = argparse.ArgumentParser()
add_precision_arguments(parser)
args = parser.parse_args()
_, _, dtype = precision_dtypes(args)
plotfile = Path("diags/diag000001")
with (plotfile / "Header").open() as header:
    header.readline()
    names = [header.readline().strip() for _ in range(int(header.readline()))]
fields = _read_buffer(str(plotfile), str(plotfile / "Level_0/Cell_H"), names)
energy = np.atleast_2d(np.loadtxt("diags/particle_energy.txt"))
rtol = 3.0e-5 if dtype == np.float32 else 8.0e-13
atol = 3.0e-6 if dtype == np.float32 else 2.0e-12
np.testing.assert_array_equal(fields["radiation_diffusion_energy"], 0)
np.testing.assert_allclose(
    np.asarray(fields["radiation_material_energy"]).reshape(-1) / energy[0, 2],
    [0, -np.expm1(-0.2), 0, 0],
    rtol=rtol,
    atol=atol,
)
np.testing.assert_allclose(energy[-1, 2] / energy[0, 2], np.exp(-0.2), rtol=rtol)
print(
    "Zero residence causes no packet-to-diffusion conversion; thin-side attenuation passes."
)
