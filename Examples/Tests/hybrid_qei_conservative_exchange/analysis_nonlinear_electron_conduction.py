#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL

"""Finite nonlinear heat redistribution: independent continuous-time EOS/PDE."""

import argparse
import subprocess

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags


def reference(cells, table):
    base = 1.0e4 if table else 10 * 1.602176634e-19 / 1.380649e-23
    initial = base * (1 + 0.5 * np.cos(2 * np.pi * np.arange(cells) / cells))
    kappa = 1.2e8 if table else 0.5

    def rhs(_, temperature):
        if table:
            capacity = 1000 * (2.0e4 + 10 * temperature)
        else:
            power = (temperature / base) ** 8
            latent_density = 0.5 * (2 / 3) / 1.380649e-23 * 40 * 1.602176634e-19
            capacity = 0.5 + latent_density * 8 * power / (
                temperature * (1 + power) ** 2
            )
        laplacian = (
            np.roll(temperature, 1) - 2 * temperature + np.roll(temperature, -1)
        ) * cells**2
        return kappa * laplacian / capacity

    sparsity = diags(
        [np.ones(cells - 1), np.ones(cells), np.ones(cells - 1)],
        [-1, 0, 1],
        format="lil",
    )
    sparsity[0, -1] = sparsity[-1, 0] = 1
    solution = solve_ivp(
        rhs,
        [0, 0.03],
        initial,
        method="BDF",
        rtol=1.0e-10,
        atol=1.0e-8,
        jac_sparsity=sparsity.tocsr(),
    )
    assert solution.success, solution.message
    return solution.y[:, -1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("executable")
    parser.add_argument("--table-file")
    parser.add_argument("--mpi-exec")
    args = parser.parse_args()
    command = [args.executable, "amrex.the_arena_init_size=0"]
    if args.table_file:
        command.append(f"test.table_file={args.table_file}")
    if args.mpi_exec:
        command = [args.mpi_exec, "-n", "2", *command]
    subprocess.run(command, check=True)
    data = np.loadtxt("nonlinear_conduction.csv", delimiter=",", skiprows=1)
    assert data.shape == (64, 5) and np.all(np.isfinite(data))
    coarse = reference(256, bool(args.table_file))[::4]
    fine = reference(512, bool(args.table_file))[::8]
    scale = np.linalg.norm(fine - data[:, 1])
    assert scale > 0.05 * np.linalg.norm(data[:, 1] - data[:, 1].mean())
    refinement = np.linalg.norm(coarse - fine) / scale
    error = np.linalg.norm(data[:, 2] - fine) / scale
    assert refinement < 0.002, refinement
    assert error < 0.03, error
    assert data[:, 2].min() >= data[:, 1].min() and data[:, 2].max() <= data[:, 1].max()
    energy_error = abs(np.sum(data[:, 4] - data[:, 3])) / np.sum(
        abs(data[:, 3]) + abs(data[:, 4])
    )
    assert energy_error < 1.0e-10, energy_error
    print(
        f"Nonlinear conduction: reference refinement={refinement:.6g}, profile error={error:.6g}, energy={energy_error:.6g}"
    )
