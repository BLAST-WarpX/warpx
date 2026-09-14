#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Native PIC moving gray pulse: independent transport and actual inventories.

This high-inertia case qualifies dynamic diffusion, not a radiation-dominated
acceleration trajectory. Inventory gates report the arithmetic conditioning
from the much larger material background instead of hiding it.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt

yt.set_log_level(50)
C = np.longdouble("299792458")
MP = np.longdouble("1.67262192595e-27")
QE = np.longdouble("1.602176634e-19")
KB = np.longdouble("1.380649e-23")


def plotfiles(directory):
    """Current outputs only; AMReX preserves overwritten plots as .old.*."""
    return sorted(
        (
            path
            for path in (Path(directory) / "diags").glob("diag1[0-9]*")
            if path.is_dir() and path.name[5:].isascii() and path.name[5:].isdigit()
        ),
        key=lambda path: int(path.name[5:]),
    )


def state(path):
    ds = yt.load(str(path))
    assert ds.dimensionality == 1
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    n = int(ds.domain_dimensions[0])
    width = np.longdouble(ds.domain_width[0].v)
    data = ds.all_data()
    order = np.lexsort((data["ions", "particle_cpu"].v, data["ions", "particle_id"].v))

    def p(name):
        return np.asarray(data["ions", f"particle_{name}"].v, dtype=np.longdouble)[
            order
        ]

    result = {
        "n": n,
        "dx": width / n,
        "time": float(ds.current_time.v),
        "ids": np.stack((p("id"), p("cpu")), axis=1),
        "weight": p("weight"),
        "position": p("position_x"),
        "u": np.stack([p(f"momentum_{axis}") / MP for axis in "xyz"], axis=1),
        "carry": np.stack(
            [p(f"radiation_impulse_diffusion_0_u{axis}") for axis in "xyz"], axis=1
        ),
        "carry_work": p("radiation_impulse_diffusion_0_work"),
    }
    for name in (
        "rho",
        "Te",
        "radiation_diffusion_energy",
        "radiation_moment_qx",
        "radiation_moment_qy",
        "radiation_moment_qz",
    ):
        result[name] = np.asarray(grid["boxlib", name].v, dtype=np.longdouble).ravel()
    assert len(order) == 4 * n
    for value in result.values():
        assert np.isfinite(value).all()
    return result


def check(directory, reference_directory=None):
    directory = Path(directory)
    plots = plotfiles(directory)
    reference_plots = (
        plots if reference_directory is None else plotfiles(reference_directory)
    )
    old, new = state(reference_plots[0]), state(plots[-1])
    assert old["time"] == 0
    assert abs(new["time"] / (1e-8 / 3) - 1) < 1e-12
    assert np.array_equal(old["ids"], new["ids"])
    assert np.array_equal(old["weight"], new["weight"])
    initial, final = (
        old["radiation_diffusion_energy"],
        new["radiation_diffusion_energy"],
    )
    n, dx = old["n"], float(old["dx"])
    time, velocity = new["time"], 9e4
    wave = 2 * np.pi * np.fft.rfftfreq(n, dx)
    transform = np.fft.rfft(np.asarray(initial, dtype=float))
    reference = np.fft.irfft(
        transform
        * np.exp(-float(C) / 3e7 * wave**2 * time - 1j * wave * velocity * time),
        n=n,
    )
    transformed_final = np.fft.rfft(np.asarray(final, dtype=float))
    displacement = -np.angle(transformed_final[1] / transform[1]) / wave[1]
    gamma0 = np.sqrt(1 + np.sum(old["u"] ** 2, axis=1) / C**2)
    gamma1 = np.sqrt(1 + np.sum(new["u"] ** 2, axis=1) / C**2)
    mass = MP * old["weight"]
    du = new["u"] - old["u"]
    kinetic = np.sum(
        mass * np.sum(du * (new["u"] + old["u"]), axis=1) / (gamma0 + gamma1)
    )
    internal = np.sum(
        (new["rho"] - old["rho"]) * old["Te"] + new["rho"] * (new["Te"] - old["Te"])
    ) * (1.5 * KB / QE * dx)
    carry = np.sum(mass * (new["carry_work"] - old["carry_work"]))
    energy_error = np.sum(final - initial) + kinetic + internal + carry
    background = np.sum(mass * np.sum(old["u"] ** 2, axis=1) / (gamma0 + 1))
    background += np.sum(old["rho"] * old["Te"]) * (1.5 * KB / QE * dx)
    arithmetic = 128 * np.finfo(float).eps * (background + np.sum(initial))
    energy_bound = 1e-10 * np.sum(initial) + arithmetic
    momentum_error = []
    momentum_bound = []
    for d, axis in enumerate("xyz"):
        delta = (
            np.sum(new[f"radiation_moment_q{axis}"] - old[f"radiation_moment_q{axis}"])
            / C
        )
        delta += np.sum(mass * (du[:, d] + new["carry"][:, d] - old["carry"][:, d]))
        bound = 1e-10 * np.sum(initial) / C + 128 * np.finfo(float).eps * np.sum(
            mass * abs(old["u"][:, d])
        )
        momentum_error.append(float(delta))
        momentum_bound.append(float(bound))
        assert abs(delta) <= bound, (delta, bound)
    result = {
        "cells": n,
        "time_s": time,
        "profile_relative_l1": float(np.sum(abs(final - reference)) / np.sum(initial)),
        "displacement_m": float(displacement),
        "expected_displacement_m": velocity * time,
        "phase_error_domain_fraction": float(
            abs(displacement - velocity * time) / (n * dx)
        ),
        "radiation_inventory_relative_change": float(
            np.sum(final - initial) / np.sum(initial)
        ),
        "density_max_relative_change": float(np.max(abs(new["rho"] / old["rho"] - 1))),
        "velocity_max_relative_change": float(
            np.max(abs(new["u"][:, 2] / gamma1 / velocity - 1))
        ),
        "actual_kinetic_change_J": float(kinetic),
        "electron_internal_change_J": float(internal),
        "particle_carry_change_J": float(carry),
        "raw_energy_error_J": float(energy_error),
        "energy_arithmetic_bound_J": float(arithmetic),
        "energy_gate_bound_J": float(energy_bound),
        "raw_momentum_error_kg_m_s": momentum_error,
        "momentum_gate_bounds_kg_m_s": momentum_bound,
    }
    assert result["profile_relative_l1"] < 0.02, result
    assert result["phase_error_domain_fraction"] < 0.002, result
    assert abs(result["radiation_inventory_relative_change"]) < 1e-4, result
    assert result["density_max_relative_change"] < 1e-4, result
    assert result["velocity_max_relative_change"] < 1e-4, result
    assert abs(energy_error) <= energy_bound, result
    # Source-only ledger is independently tight even with a large material
    # background; numerical residuals are measured, never used as corrections.
    ledger = np.loadtxt(directory / "diags" / "radiation_energy.txt")
    assert ledger.shape[1] == 19
    assert np.max(
        abs(ledger[:, 2] + ledger[:, 6] + ledger[:, 16] - np.sum(initial))
    ) < 1e-10 * np.sum(initial)
    if reference_directory is not None:
        reference_state = state(reference_plots[-1])
        assert np.array_equal(new["ids"], reference_state["ids"])
        assert (
            np.max(abs(new["position"] - reference_state["position"])) < 1e-11 * n * dx
        )
        assert np.max(abs(new["u"] - reference_state["u"])) < 1e-12 * velocity
        for name in (
            "rho",
            "Te",
            "radiation_diffusion_energy",
            "radiation_moment_qx",
            "radiation_moment_qy",
            "radiation_moment_qz",
        ):
            scale = np.max(abs(reference_state[name]))
            assert np.max(abs(new[name] - reference_state[name])) <= 1e-10 * scale
        result["changed_rank_restart_matches"] = True
    x = (np.arange(n) + 0.5) * dx
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(x, initial / dx, "--", label="Initial radiation")
    axes[0].plot(x, reference / dx, label="Independent thick-limit solution")
    axes[0].plot(x, final / dx, ":", linewidth=2, label="WarpX native PIC + coupled M1")
    axes[0].set_ylabel("Radiation energy [J/m^3]")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, (final - reference) / max(initial))
    axes[1].set_ylabel("Error / initial peak")
    axes[1].set_xlabel("z [m]")
    fig.suptitle(f"Native moving pulse: {n} cells, t={time:.4g} s")
    fig.tight_layout()
    fig.savefig(directory / "pulse.png", dpi=160)
    plt.close(fig)
    (directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--reference")
    args = parser.parse_args()
    check(args.directory, args.reference)
