#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Cylindrical nonlinear conduction with the measured native nodal density."""

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags


def reference(data, cells):
    x = np.linspace(0, 1, cells + 1)
    coarse_x = data[:, 0] / data[-1, 0]
    temperature = np.interp(x, coarse_x, data[:, 1])
    number = np.interp(x, coarse_x, data[:, 3]) / 1.602176634e-19
    mass_density = np.interp(x, coarse_x, data[:, 7])
    table = bool(np.any(mass_density > 0))
    kappa = 1.2e8 if table else 0.5
    volume = np.pi * (
        np.minimum(1, x + 0.5 / cells) ** 2 - np.maximum(0, x - 0.5 / cells) ** 2
    )
    face_area = 2 * np.pi * (np.arange(cells) + 0.5) / cells
    transition = 10 * 1.602176634e-19 / 1.380649e-23

    def rhs(_, t):
        if table:
            capacity = mass_density * (2e4 + 10 * t)
        else:
            power = (t / transition) ** 8
            capacity = number * (
                1.380649e-23 / (2 / 3)
                + 40 * 1.602176634e-19 * 8 * power / (t * (1 + power) ** 2)
            )
        flux = face_area * kappa * np.diff(t) * cells
        return np.diff(np.r_[0, flux, 0]) / (volume * capacity)

    sparsity = diags([np.ones(cells), np.ones(cells + 1), np.ones(cells)], [-1, 0, 1])
    solution = solve_ivp(
        rhs,
        [0, 0.03],
        temperature,
        method="BDF",
        rtol=1e-10,
        atol=1e-8,
        jac_sparsity=sparsity.tocsr(),
    )
    assert solution.success, solution.message
    return np.interp(coarse_x, x, solution.y[:, -1])


def check(directory):
    data = np.loadtxt(
        Path(directory) / "native_nonlinear_conduction.csv", delimiter=",", skiprows=1
    )
    assert data.shape == (65, 9) and np.all(np.isfinite(data))
    assert np.all(data[:, 3] > 0) and np.all(data[:, 8] > 0)
    np.testing.assert_allclose(data[:, 3], data[:, 4], rtol=1e-12)
    # Do not assume the PIC deposition equals a spatially constant fluid density.
    coarse, fine = reference(data, 256), reference(data, 512)
    signal = np.linalg.norm(fine - data[:, 1])
    assert signal > 0.05 * np.linalg.norm(data[:, 1] - data[:, 1].mean())
    refinement = np.linalg.norm(fine - coarse) / signal
    error = np.linalg.norm(data[:, 2] - fine) / signal
    assert refinement < 0.002, refinement
    assert error < 0.03, error
    assert data[:, 2].min() >= data[:, 1].min() and data[:, 2].max() <= data[:, 1].max()
    energy_error = abs(np.sum(data[:, 8] * (data[:, 6] - data[:, 5]))) / np.sum(
        data[:, 8] * (abs(data[:, 5]) + abs(data[:, 6]))
    )
    assert energy_error < 1e-10, energy_error
    print(
        f"Native RZ nonlinear conduction: profile={error:.6g}, refinement={refinement:.6g}, energy={energy_error:.6g}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    check(parser.parse_args().directory)
