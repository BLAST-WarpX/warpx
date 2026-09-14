#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL

"""Prescribed radial compression: independent nonrelativistic diffusion/work PDE.

The material velocity is prescribed, not evolved PIC matter. This checks the
new cylindrical radiation operator, not the guarded native moving-material API.
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply


def reference(cells):
    faces = np.linspace(0.0, 1.0, cells + 1)
    dr = 1.0 / cells
    volume = np.pi * np.diff(faces**2)
    area = 2.0 * np.pi * faces
    velocity = -1.0e-3 * np.sin(np.pi * faces)
    velocity[[0, -1]] = 0.0
    divergence = np.diff(area * velocity) / volume
    diagonal = -divergence / 3.0
    lower = np.zeros(cells - 1)
    upper = np.zeros(cells - 1)
    diffusion = 1.0 / 3000.0  # Time coordinate is xi=c*t.
    for face in range(1, cells):
        left = area[face] * (max(velocity[face], 0.0) + diffusion / dr)
        right = area[face] * (min(velocity[face], 0.0) - diffusion / dr)
        diagonal[face - 1] -= left / volume[face - 1]
        upper[face - 1] -= right / volume[face - 1]
        lower[face - 1] += left / volume[face]
        diagonal[face] += right / volume[face]
    operator = diags([lower, diagonal, upper], [-1, 0, 1], format="csr")
    # E_xi = -div(beta*E) - E*div(beta)/3 + Laplacian(E)/(3*kappa).
    np.testing.assert_allclose(
        operator @ np.ones(cells), -4 * divergence / 3, atol=1.0e-10
    )
    np.testing.assert_allclose(
        volume @ operator, -volume * divergence / 3, atol=1.0e-12
    )
    result = expm_multiply(
        30.0 * operator, np.ones(cells), traceA=30.0 * diagonal.sum()
    )
    assert np.all(np.isfinite(result)) and np.all(result > 0)
    return result, volume


parser = argparse.ArgumentParser()
parser.add_argument("executable", type=Path)
args = parser.parse_args()
coarse_reference, coarse_volume = reference(512)
fine_reference, fine_volume = reference(1024)
restricted = (fine_reference * fine_volume).reshape(512, 2).sum(axis=1) / coarse_volume
reference_error = np.sqrt(
    np.sum(coarse_volume * (restricted - coarse_reference) ** 2)
    / np.sum(coarse_volume * (restricted - 1.0) ** 2)
)
assert reference_error < 0.002, reference_error

subprocess.run(
    [
        str(args.executable.resolve()),
        "test.case=compression",
        "test.nr=32",
        "amrex.the_arena_init_size=0",
    ],
    check=True,
)
data = np.loadtxt("rz_m1_compression.csv", delimiter=",", skiprows=1)
assert data.shape == (32, 6) and np.all(np.isfinite(data))
volume = np.pi * (data[:, 1] ** 2 - data[:, 0] ** 2)
expected = (fine_reference * fine_volume).reshape(32, 32).sum(axis=1) / volume

# Transform the measured cell moments back to the material frame, using the
# standard M1 Eddington factor independently of the C++ geometry implementation.
energy = data[:, 2]
flux = data[:, 3:6]
assert np.all(energy > 0) and np.all(flux[:, 1] == 0)
f2 = np.sum(flux**2, axis=1) / energy**2
assert np.all(f2 <= 1.0)
chi = (3 + 4 * f2) / (5 + 2 * np.sqrt(4 - 3 * f2))
radial_fraction = np.divide(
    flux[:, 0] ** 2, np.sum(flux**2, axis=1), out=np.zeros(32), where=f2 > 0
)
pressure = energy * ((1 - chi) / 2 + (3 * chi - 1) * radial_fraction / 2)
beta = -1.0e-3 * np.sin(np.pi * (data[:, 0] + data[:, 1]) / 2)
measured = (energy - 2 * beta * flux[:, 0] + beta**2 * pressure) / (1 - beta**2)
error = np.sqrt(
    np.sum(volume * (measured - expected) ** 2) / np.sum(volume * (expected - 1) ** 2)
)
assert expected.max() > 1.2 and expected.min() < 0.9
print(
    f"Radial compression: reference refinement={reference_error:.6g}, M1 profile error={error:.6g}"
)
assert error < 0.03, error
