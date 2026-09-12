#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Check the runtime implicit path against its backward-Euler Fourier mode."""

import re
from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer


def read(path):
    with (path / "Header").open() as header:
        header.readline()
        names = [header.readline().strip() for _ in range(int(header.readline()))]
    return _read_buffer(str(path), str(path / "Level_0/Cell_H"), names)


initial = read(Path("diags/diag1000000"))
final = read(Path("diags/diag1000001"))
e0 = np.asarray(initial["radiation_diffusion_energy"]).squeeze()
e = np.asarray(final["radiation_diffusion_energy"]).squeeze()
ledger_path = Path("diags/radiation_energy.txt")
labels = re.findall(r"\[\d+\]([^\s]+)", ledger_path.read_text().splitlines()[0])
columns = {name: i for i, name in enumerate(labels)}
ledger = np.atleast_2d(np.loadtxt(ledger_path))
cells = e.shape[0]
dimensions = e.ndim
dx = 1 / cells
volume = dx**dimensions
dt = ledger[-1, columns["time(s)"]]
eigenvalue = 4 * np.sin(np.pi / cells) ** 2 / dx**2
expected_amplitude = 0.01 / dimensions / (1 + dt * 299792458 / 300 * eigenvalue)
assert e.shape in ((32,), (16, 16, 16))
assert np.all(np.isfinite(e)) and np.all(e >= 0)
for axis in range(dimensions):
    shape = [1] * dimensions
    shape[axis] = cells
    mode = np.cos(2 * np.pi * (np.arange(cells) + 0.5) * dx).reshape(shape)
    amplitude = 2 * np.sum((e - volume) * mode)
    np.testing.assert_allclose(amplitude, expected_amplitude, rtol=2e-6)
np.testing.assert_allclose(e.sum(), e0.sum(), rtol=1e-10)
assert ledger[-1, columns["diffusion_relative_residual()"]] <= 1e-10
assert ledger[-1, columns["diffusion_nonlinear_iterations()"]] > 0
assert ledger[-1, columns["diffusion_linear_iterations()"]] > 0
print(
    "Runtime implicit FLD: backward-Euler mode, positivity, conservation and residual pass"
)
