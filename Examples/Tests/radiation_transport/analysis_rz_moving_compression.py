#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Finite radial compression/LTE: actual motion, heat, recoil and ownership.

This is a nonlinear native integration gate, not an exact FLASH-fluid trajectory
comparison. The separate axial pulse supplies a transport-accuracy reference.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analysis_rz_moving_moment import MP, C, plotfiles, state


def check(directory, reference_directory=None):
    directory = Path(directory)
    original = directory if reference_directory is None else Path(reference_directory)
    old, new = state(plotfiles(original)[0]), state(plotfiles(directory)[-1])
    assert old["time"] == 0 and abs(new["time"] / 2.56e-9 - 1) < 1e-12
    assert np.array_equal(old["ids"], new["ids"])
    assert np.array_equal(old["weight"], new["weight"])
    nr, nz = old["rho"].shape
    radius = 1e-3
    dr = radius / nr
    mass = MP * old["weight"]
    g0 = np.sqrt(1 + np.sum(old["u"] ** 2, axis=1) / C**2)
    g1 = np.sqrt(1 + np.sum(new["u"] ** 2, axis=1) / C**2)
    du = new["u"] - old["u"]
    work = np.sum(mass * np.sum(du * (new["u"] + old["u"]), axis=1) / (g0 + g1))
    carry = np.sum(mass * (new["carry_work"] - old["carry_work"]))
    before = np.loadtxt(original / "native_inventory.txt", dtype=np.longdouble)[0]
    after = np.loadtxt(directory / "native_inventory.txt", dtype=np.longdouble)[1]
    radiation_change = np.sum(
        new["radiation_diffusion_energy"] - old["radiation_diffusion_energy"]
    )
    electron_change = after[1] - before[1]
    energy_error = radiation_change + electron_change + work + carry
    arithmetic = 128 * np.finfo(float).eps * np.sum(abs(before))
    energy_bound = 1e-10 * before[2] + arithmetic
    displacement = new["r"] - old["r"]
    crossed = np.floor(new["r"] / dr) != np.floor(old["r"] / dr)
    diagnostic = directory / "diags" / "radiation_momentum.txt"
    columns = {
        name: int(index)
        for index, name in re.findall(
            r"\[(\d+)\]([^\s(]+)", diagnostic.read_text().splitlines()[0]
        )
    }
    ledger = np.atleast_2d(np.loadtxt(diagnostic))[-1]
    recoil = ledger[columns["cumulative_material_r"]]
    boundary = ledger[columns["cumulative_moment_boundary_r"]]
    geometric = ledger[columns["cumulative_moment_geometric_r"]]
    balance = (
        ledger[columns["moment_radiation_r"]]
        + recoil
        + ledger[columns["pending_diffusion_material_r"]]
        + ledger[columns["cumulative_moment_boundary_minus_geometric_r"]]
    )
    result = {
        "radial_cells": nr,
        "axial_cells": nz,
        "time_s": new["time"],
        "maximum_inward_displacement_m": float(-displacement.min()),
        "fraction_particles_crossing_radial_cells": float(np.mean(crossed)),
        "actual_kinetic_change_J": float(work),
        "electron_internal_change_J": float(electron_change),
        "radiation_change_J": float(radiation_change),
        "raw_energy_error_J": float(energy_error),
        "energy_bound_J": float(energy_bound),
        "density_max_relative_change": float(np.max(abs(new["rho"] / old["rho"] - 1))),
        "radiation_material_radial_impulse_kg_m_s": float(recoil),
        "optical_wall_radial_impulse_kg_m_s": float(boundary),
        "geometric_radial_contribution_kg_m_s": float(geometric),
        "source_radial_balance_kg_m_s": float(balance),
        "gates_passed": False,
    }
    (directory / "rz_moving_compression.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    # Bound finite effects, not just a small residual for a nearly static state.
    assert -displacement.min() > dr, result
    assert np.mean(crossed) > 0.25, result
    assert result["density_max_relative_change"] > 0.1, result
    assert electron_change > 0.25 * before[1], result
    assert radiation_change < -0.4 * before[2], result
    assert work < -0.01 * before[0], result
    assert abs(work - (after[0] - before[0])) <= arithmetic, result
    assert abs(energy_error) <= energy_bound, result
    assert recoil > 1e-2 * before[2] / C, result
    assert boundary > 0 and geometric > boundary, result
    assert abs(balance) <= 1e-10 * before[2] / C, result
    if reference_directory is not None:
        uninterrupted = state(plotfiles(original)[-1])
        assert np.array_equal(new["ids"], uninterrupted["ids"])
        assert np.max(abs(new["r"] - uninterrupted["r"])) < 1e-11 * radius
        assert np.max(abs(new["z"] - uninterrupted["z"])) < 1e-11 * radius
        assert np.max(abs(new["u"] - uninterrupted["u"])) < 1e-12 * 2e4
        for name in (
            "rho",
            "Te",
            "radiation_diffusion_energy",
            "radiation_moment_qx",
            "radiation_moment_qy",
            "radiation_moment_qz",
        ):
            scale = np.max(abs(uninterrupted[name]))
            rounding = 0
            if name.startswith("radiation_moment"):
                rounding = (
                    128
                    * np.finfo(float).eps
                    * np.max(abs(uninterrupted["radiation_diffusion_energy"]))
                )
            assert (
                np.max(abs(new[name] - uninterrupted[name])) <= 1e-10 * scale + rounding
            )
        old_ledger = np.atleast_2d(
            np.loadtxt(original / "diags" / "radiation_momentum.txt")
        )[-1]
        for name, column in columns.items():
            if name.startswith("cumulative_moment_"):
                scale = before[2] if name.endswith("energy") else before[2] / C
                assert abs(ledger[column] - old_ledger[column]) <= (
                    1e-10 * scale + 128 * np.finfo(float).eps * abs(old_ledger[column])
                )
        result["restart_matches"] = True
    result["gates_passed"] = True
    (directory / "rz_moving_compression.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    r = (np.arange(nr) + 0.5) * dr
    volume = np.pi * ((r + dr / 2) ** 2 - (r - dr / 2) ** 2) * old["length"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for snapshot, label in ((old, "initial"), (new, "final")):
        axes[0].plot(
            r * 1e3,
            np.asarray(snapshot["rho"].mean(axis=1) / old["rho"].mean(), dtype=float),
            label=label,
        )
        axes[1].plot(
            r * 1e3,
            np.asarray(snapshot["Te"].mean(axis=1) / old["Te"].mean(), dtype=float),
            label=label,
        )
        axes[2].plot(
            r * 1e3,
            np.asarray(
                snapshot["radiation_diffusion_energy"].sum(axis=1) / volume / 1e11,
                dtype=float,
            ),
            label=label,
        )
    for axis, label in zip(
        axes,
        (
            "density / initial mean",
            "electron T / initial mean",
            "radiation density / initial",
        ),
    ):
        axis.set(xlabel="r (mm)", ylabel=label)
        axis.legend(fontsize=8)
    figure.savefig(directory / "rz_moving_compression.png", dpi=160)
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference-directory", type=Path)
    args = parser.parse_args()
    check(args.directory, args.reference_directory)
