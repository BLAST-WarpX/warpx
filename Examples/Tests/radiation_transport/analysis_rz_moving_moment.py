#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Moving native RZ pulse: particle motion/work, annular transport and restart.

The linear trapped-pulse reference tests radiation transport only. Kinetic PIC
ions are not required to reproduce a FLASH fluid-ion closure. Native nodal
electron energy is measured by the C++ driver, not inferred from cell-averaged Te.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt
from analysis_moving_moment_pulse import MP, C, plotfiles

yt.set_log_level(50)


def state(path):
    ds = yt.load(str(path))
    assert ds.dimensionality == 2 and float(ds.domain_left_edge[0].v) == 0
    data = ds.all_data()
    order = np.lexsort((data["ions", "particle_cpu"].v, data["ions", "particle_id"].v))

    def particle(name):
        return np.asarray(data["ions", "particle_" + name].v, dtype=np.longdouble)[
            order
        ]

    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    result = {
        "time": float(ds.current_time.v),
        "length": float(ds.domain_width[1].v),
        "ids": np.stack((particle("id"), particle("cpu")), axis=1),
        "weight": particle("weight"),
        # WarpX RZ plotfiles label the two stored coordinates x=r and y=z.
        "r": particle("position_x"),
        "z": particle("position_y"),
        "u": np.stack([particle("momentum_" + a) / MP for a in "xyz"], axis=1),
        "carry": np.stack(
            [particle("radiation_impulse_diffusion_0_u" + a) for a in "xyz"], axis=1
        ),
        "carry_work": particle("radiation_impulse_diffusion_0_work"),
    }
    for name in (
        "rho",
        "Te",
        "radiation_diffusion_energy",
        "radiation_moment_qx",
        "radiation_moment_qy",
        "radiation_moment_qz",
    ):
        result[name] = np.asarray(grid["boxlib", name].v, dtype=np.longdouble).squeeze()
    for value in result.values():
        assert np.isfinite(value).all()
    return result


def check(directory, reference_directory=None, report_only=False):
    directory = Path(directory)
    original = directory if reference_directory is None else Path(reference_directory)
    old, new = state(plotfiles(original)[0]), state(plotfiles(directory)[-1])
    assert old["time"] == 0 and abs(new["time"] / 2.56e-9 - 1) < 1e-12
    assert np.array_equal(old["ids"], new["ids"])
    assert np.array_equal(old["weight"], new["weight"])
    length, time = old["length"], new["time"]
    mass = MP * old["weight"]
    g0 = np.sqrt(1 + np.sum(old["u"] ** 2, axis=1) / C**2)
    g1 = np.sqrt(1 + np.sum(new["u"] ** 2, axis=1) / C**2)
    du = new["u"] - old["u"]
    work = np.sum(mass * np.sum(du * (new["u"] + old["u"]), axis=1) / (g0 + g1))
    carry = np.sum(mass * (new["carry_work"] - old["carry_work"]))
    displacement = (new["z"] - old["z"] + length / 2) % length - length / 2
    before = np.loadtxt(original / "native_inventory.txt", dtype=np.longdouble)[0]
    after = np.loadtxt(directory / "native_inventory.txt", dtype=np.longdouble)[1]
    radiation_change = np.sum(
        new["radiation_diffusion_energy"] - old["radiation_diffusion_energy"]
    )
    energy_error = radiation_change + work + after[1] - before[1] + carry
    arithmetic = 128 * np.finfo(float).eps * np.sum(np.abs(before))
    energy_bound = 1e-10 * before[2] + arithmetic
    momentum_error = np.sum(new["radiation_moment_qz"] - old["radiation_moment_qz"]) / C
    momentum_error += np.sum(
        mass * (du[:, 2] + new["carry"][:, 2] - old["carry"][:, 2])
    )
    momentum_bound = 1e-10 * before[2] / C + 128 * np.finfo(float).eps * np.sum(
        mass * abs(old["u"][:, 2])
    )

    # Summing integrated energy over annuli gives the physical axial inventory,
    # without treating yt's Cartesian geometry label as a cylindrical measure.
    initial = np.asarray(old["radiation_diffusion_energy"].sum(axis=0), dtype=float)
    final = np.asarray(new["radiation_diffusion_energy"].sum(axis=0), dtype=float)
    n = len(initial)
    wave = 2 * np.pi * np.fft.rfftfreq(n, length / n)
    transform = np.fft.rfft(initial)
    reference = np.fft.irfft(
        transform * np.exp(-float(C) / 3e7 * wave**2 * time - 1j * wave * 1e5 * time),
        n=n,
    )
    phase = -np.angle(np.fft.rfft(final)[1] / transform[1]) / wave[1]
    diagnostic = directory / "diags" / "radiation_momentum.txt"
    columns = {
        name: int(index)
        for index, name in re.findall(
            r"\[(\d+)\]([^\s(]+)", diagnostic.read_text().splitlines()[0]
        )
    }
    ledger = np.atleast_2d(np.loadtxt(diagnostic))[-1]
    net_radial = ledger[columns["cumulative_moment_boundary_minus_geometric_r"]]
    # The source-stage radial inventory is NOT global material momentum: native
    # pressure forces and material-wall forces have their own ownership.
    source_radial = (
        ledger[columns["moment_radiation_r"]]
        + ledger[columns["cumulative_material_r"]]
        + ledger[columns["pending_diffusion_material_r"]]
        + net_radial
    )
    result = {
        "axial_cells": n,
        "time_s": time,
        "particle_displacement_min_m": float(displacement.min()),
        "particle_displacement_max_m": float(displacement.max()),
        "actual_kinetic_change_J": float(work),
        "electron_internal_change_J": float(after[1] - before[1]),
        "radiation_change_J": float(radiation_change),
        "raw_energy_error_J": float(energy_error),
        "energy_bound_J": float(energy_bound),
        "raw_axial_momentum_error_kg_m_s": float(momentum_error),
        "axial_momentum_bound_kg_m_s": float(momentum_bound),
        "source_radial_balance_kg_m_s": float(source_radial),
        "profile_relative_l1": float(np.sum(abs(final - reference)) / np.sum(initial)),
        "phase_error_domain_fraction": float(abs(phase - 1e5 * time) / length),
        "density_max_relative_change": float(np.max(abs(new["rho"] / old["rho"] - 1))),
        "velocity_max_relative_change": float(
            np.max(abs(new["u"][:, 2] / g1 / 1e5 - 1))
        ),
        "qualification_enforced": not report_only,
        "gates_passed": False,
    }
    (directory / "rz_moving_moment.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    z = (np.arange(n) + 0.5) * length / n
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(z * 1e3, initial / initial.mean(), label="initial")
    axes[0].plot(
        z * 1e3,
        reference / initial.mean(),
        "--",
        label="linear trapped-pulse reference",
    )
    axes[0].plot(z * 1e3, final / initial.mean(), label="native moving RZ")
    axes[0].set(xlabel="z (mm)", ylabel="axial radiation energy / initial mean")
    axes[0].legend(fontsize=8)
    axes[1].scatter(
        np.asarray(new["z"], dtype=float) * 1e3,
        np.asarray(new["u"][:, 2] / g1 - 1e5, dtype=float),
        s=1,
    )
    axes[1].set(xlabel="particle z (mm)", ylabel="ion axial velocity change (m/s)")
    figure.savefig(directory / "rz_moving_moment.png", dpi=160)
    plt.close(figure)
    if not report_only:
        assert np.min(displacement) > 0.2 * length, result
        assert result["density_max_relative_change"] > 1e-4, result
        assert result["velocity_max_relative_change"] > 1e-4, result
        assert abs(work) > 1e-8 * before[0], result
        assert abs(work - (after[0] - before[0])) <= arithmetic, result
        assert abs(energy_error) <= energy_bound, result
        assert abs(momentum_error) <= momentum_bound, result
        assert abs(source_radial) <= 1e-10 * before[2] / C, result
        assert result["profile_relative_l1"] < 0.02, result
        assert result["phase_error_domain_fraction"] < 0.002, result
        if reference_directory is not None:
            uninterrupted = state(plotfiles(original)[-1])
            assert np.array_equal(new["ids"], uninterrupted["ids"])
            assert np.array_equal(new["weight"], uninterrupted["weight"])
            for coordinate in ("r", "z"):
                assert (
                    np.max(abs(new[coordinate] - uninterrupted[coordinate]))
                    < 1e-11 * length
                )
            assert np.max(abs(new["u"] - uninterrupted["u"])) < 1e-12 * 1e5
            for name in (
                "rho",
                "Te",
                "radiation_diffusion_energy",
                "radiation_moment_qx",
                "radiation_moment_qy",
                "radiation_moment_qz",
            ):
                scale = np.max(abs(uninterrupted[name]))
                arithmetic_floor = 0
                if name.startswith("radiation_moment"):
                    arithmetic_floor = (
                        128
                        * np.finfo(float).eps
                        * np.max(abs(uninterrupted["radiation_diffusion_energy"]))
                    )
                assert (
                    np.max(abs(new[name] - uninterrupted[name]))
                    <= 1e-10 * scale + arithmetic_floor
                )
            old_ledger = np.atleast_2d(
                np.loadtxt(original / "diags" / "radiation_momentum.txt")
            )[-1]
            for name, column in columns.items():
                if name.startswith("cumulative_moment_"):
                    unit_scale = before[2] if name.endswith("energy") else before[2] / C
                    assert abs(ledger[column] - old_ledger[column]) <= (
                        1e-10 * unit_scale
                        + 128 * np.finfo(float).eps * abs(old_ledger[column])
                    )
            result["restart_matches"] = True
        result["gates_passed"] = True
    (directory / "rz_moving_moment.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference-directory", type=Path)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    check(args.directory, args.reference_directory, args.report_only)
