#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Actual 2D vector exchange and native caloric inventories, not FLASH closure."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt
from analysis_moving_moment_interface import caloric, nodes
from analysis_moving_moment_pulse import MP, QE, C, plotfiles

yt.set_log_level(50)


def normal_profiles(row):
    """Project the 2:1 periodic normal without interpolation or phase fitting."""
    nx, nz = row["energy"].shape
    assert nz == 2 * nx and abs(row["width"][1] / row["width"][0] - 2) < 1e-14

    def cell_average(value):
        return (
            value
            + np.roll(value, -1, axis=0)
            + np.roll(value, -1, axis=1)
            + np.roll(np.roll(value, -1, axis=0), -1, axis=1)
        ) / 4

    arrays = {
        "radiation_energy_density": row["energy"] / row["area"],
        "radiation_normal_q_density": (2 * row["q"][..., 0] + row["q"][..., 2])
        / np.sqrt(5)
        / row["area"],
        "radiation_transverse_q_density": (row["q"][..., 0] - 2 * row["q"][..., 2])
        / np.sqrt(5)
        / row["area"],
        "charge_density": cell_average(row["rho_nodes"]),
        "electron_caloric_density": cell_average(caloric(row)),
    }
    # At cell centers s/L=(2*i+j+1.5)/nz modulo one. Every phase bin
    # has exactly nx cells, so arithmetic averaging is volume conservative.
    index = (np.arange(nz)[None, :] - 2 * np.arange(nx)[:, None] - 1) % nz
    profiles = {}
    for name, array in arrays.items():
        aligned = np.take_along_axis(array, index, axis=1)
        profiles[name] = np.mean(aligned, axis=0)
        assert abs(np.mean(profiles[name]) - np.mean(array)) <= 64 * np.finfo(
            float
        ).eps * np.max(abs(array))
    profiles["s_m"] = (np.arange(nz) + 0.5) / nz * row["width"][0] * 2 / np.sqrt(5)
    return profiles, arrays


def load(path):
    ds = yt.load(str(path))
    assert ds.dimensionality == 2
    cells = np.asarray(ds.domain_dimensions[:2], dtype=int)
    area = np.prod(np.asarray(ds.domain_width[:2].v, dtype=np.longdouble) / cells)
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    data = ds.all_data()
    order = np.lexsort((data["ions", "particle_cpu"].v, data["ions", "particle_id"].v))

    def particle(name):
        return np.asarray(data["ions", "particle_" + name].v, dtype=np.longdouble)[
            order
        ]

    step = int(path.name[5:])
    row = {
        "step": step,
        "time": float(ds.current_time.v),
        "area": area,
        "width": np.asarray(ds.domain_width[:2].v, dtype=np.longdouble),
        "ids": np.stack((particle("id"), particle("cpu")), axis=1),
        "position": np.stack([particle("position_" + axis) for axis in "xy"], axis=1),
        "mass": particle("weight") * MP,
        "u": np.stack([particle("momentum_" + axis) / MP for axis in "xyz"], axis=1),
        "carry": np.stack(
            [particle("radiation_impulse_diffusion_0_u" + axis) for axis in "xyz"],
            axis=1,
        ),
        "carry_work": particle("radiation_impulse_diffusion_0_work"),
        "rho_nodes": nodes(path / "raw_fields/Level_0/rho_fp_H", cells),
        "temperature_nodes": nodes(
            path.parent
            / f"chk{step:06d}/Level_0/hybrid_electron_temperature_fp[level=0]_H",
            cells,
        ),
        "energy": np.asarray(
            grid["boxlib", "radiation_diffusion_energy"].v, dtype=np.longdouble
        ).reshape(tuple(cells)),
        "q": np.stack(
            [
                np.asarray(
                    grid["boxlib", "radiation_moment_q" + axis].v, dtype=np.longdouble
                ).reshape(tuple(cells))
                for axis in "xyz"
            ],
            axis=-1,
        ),
    }
    assert len(order) == 4 * np.prod(cells)
    for value in row.values():
        assert np.isfinite(value).all()
    assert np.min(row["temperature_nodes"]) > 0 and np.min(row["rho_nodes"]) > 0
    assert np.min(row["energy"]) > 0
    assert np.all(
        np.sqrt(np.sum(row["q"] ** 2, axis=-1))
        <= row["energy"] * (1 + 64 * np.finfo(float).eps)
    )
    assert abs(
        np.sum(row["mass"]) - np.sum(row["rho_nodes"]) * area * MP / QE
    ) < 1e-12 * np.sum(row["mass"])
    return row


def check(directory, probe=False, reference=None):
    plots = plotfiles(directory)
    originals = plots if reference is None else plotfiles(reference)
    initial = load(originals[0])
    assert initial["step"] == 0
    gamma0 = np.sqrt(1 + np.sum(initial["u"] ** 2, axis=1) / C**2)
    energy0 = np.sum(initial["energy"])
    history = []
    for plot in plots:
        current = load(plot)
        assert np.array_equal(initial["ids"], current["ids"])
        assert np.array_equal(initial["mass"], current["mass"])
        du = current["u"] - initial["u"]
        gamma = np.sqrt(1 + np.sum(current["u"] ** 2, axis=1) / C**2)
        kinetic = np.sum(
            initial["mass"]
            * np.sum(du * (current["u"] + initial["u"]), axis=1)
            / (gamma0 + gamma)
        )
        heat = np.sum(caloric(current) - caloric(initial)) * initial["area"]
        carry = np.sum(
            initial["mass"] * (current["carry_work"] - initial["carry_work"])
        )
        radiation = np.sum(current["energy"] - initial["energy"])
        momentum = np.sum(current["q"] - initial["q"], axis=(0, 1)) / C
        momentum += np.sum(
            initial["mass"][:, None] * (du + current["carry"] - initial["carry"]),
            axis=0,
        )
        bound = 1e-8 * energy0 / C + 128 * np.finfo(float).eps * np.sum(
            initial["mass"][:, None] * abs(initial["u"]), axis=0
        )
        assert np.all(abs(momentum) < bound), momentum
        assert abs(radiation + kinetic + heat + carry) < 1e-8 * energy0
        velocity = current["u"] / gamma[:, None]
        transverse = (velocity[:, 0] - 2 * velocity[:, 2]) / np.sqrt(5)
        history.append(
            {
                "step": current["step"],
                "time_s": current["time"],
                "radiation_change": float(radiation),
                "kinetic_change": float(kinetic),
                "caloric_change": float(heat),
                "carry_change": float(carry),
                "energy_error_over_initial_radiation": float(
                    abs(radiation + kinetic + heat + carry) / energy0
                ),
                "momentum_error": [float(value) for value in momentum],
                "mass_weighted_transverse_velocity_rms_over_drift": float(
                    np.sqrt(
                        np.sum(initial["mass"] * transverse**2)
                        / np.sum(initial["mass"])
                    )
                    / 9e4
                ),
                "transverse_q_l1_over_energy": float(
                    np.sum(abs(current["q"][..., 0] - 2 * current["q"][..., 2]))
                    / np.sqrt(5)
                    / np.sum(current["energy"])
                ),
            }
        )
    ledger = np.loadtxt(directory / "diags/radiation_energy.txt")
    assert (
        np.max(abs(ledger[:, 2] + ledger[:, 6] + ledger[:, 16] - energy0))
        < 1e-10 * energy0
    )
    report = {
        "probe_only": probe,
        "initial_radiation": float(energy0),
        "history": history,
    }
    if reference is not None:
        baseline = load(originals[-1])
        assert current["step"] == baseline["step"]
        assert np.array_equal(current["ids"], baseline["ids"])
        assert np.array_equal(current["mass"], baseline["mass"])
        displacement = current["position"] - baseline["position"]
        displacement = (displacement + current["width"] / 2) % current[
            "width"
        ] - current["width"] / 2
        assert np.all(abs(displacement) < 1e-11 * current["width"])
        assert np.max(abs(current["u"] - baseline["u"])) < 1e-12 * 9e4
        for name in ("energy", "q", "rho_nodes", "temperature_nodes"):
            assert np.max(abs(current[name] - baseline[name])) <= 1e-10 * np.max(
                abs(baseline[name])
            ), name
        report["changed_rank_restart_matches"] = True
    (directory / "oblique_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    profiles, arrays = normal_profiles(current)
    np.savez(directory / "oblique_profiles.npz", **profiles)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for axis, name in zip(
        axes.flat,
        (
            "radiation_energy_density",
            "charge_density",
            "radiation_normal_q_density",
            "radiation_transverse_q_density",
        ),
        strict=True,
    ):
        plot = axis.imshow(
            np.asarray(arrays[name], dtype=float).T,
            origin="lower",
            extent=(
                0,
                float(current["width"][0]) * 1e3,
                0,
                float(current["width"][1]) * 1e3,
            ),
        )
        axis.set_title(name.replace("_", " "), fontsize=9)
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("z [mm]")
        fig.colorbar(plot, ax=axis)
    fig.suptitle(
        f"Oblique interface at {current['time'] * 1e9:.4g} ns"
        + (" — setup probe only" if probe else "")
    )
    fig.savefig(directory / "oblique_fields.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    if not probe:
        assert abs(current["time"] / (1e-8 / 3) - 1) < 1e-12
        assert abs(history[-1]["kinetic_change"]) > 1e-3 * energy0
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    check(args.directory, args.probe, args.reference)
