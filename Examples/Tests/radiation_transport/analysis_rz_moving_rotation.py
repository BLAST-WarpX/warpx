#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Independent native RZ drift/rotation inventories; not a fluid-closure oracle."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yt
from analysis_rz_moving_moment import MP, C, plotfiles, state
from matplotlib import pyplot as plt


def snapshot(path):
    result = state(path)
    data = yt.load(str(path)).all_data()
    order = np.lexsort((data["ions", "particle_cpu"].v, data["ions", "particle_id"].v))
    result["theta"] = np.asarray(data["ions", "particle_theta"].v, dtype=np.longdouble)[
        order
    ]
    return result


def angular_inventory(s):
    nr = s["rho"].shape[0]
    dr = np.longdouble("0.001") / nr
    rmid = (np.arange(nr, dtype=np.longdouble) + 0.5) * dr
    mean_radius = rmid + dr**2 / (12 * rmid)
    radiation = np.sum(mean_radius[:, None] * s["radiation_moment_qy"]) / C
    lever = MP * s["weight"] * s["r"]
    sine, cosine = np.sin(s["theta"]), np.cos(s["theta"])
    represented = np.sum(lever * (-sine * s["u"][:, 0] + cosine * s["u"][:, 1]))
    pending = np.sum(lever * (-sine * s["carry"][:, 0] + cosine * s["carry"][:, 1]))
    return np.array([radiation, represented, pending], dtype=np.longdouble)


def check(directory, reference_directory=None):
    directory = Path(directory)
    original = directory if reference_directory is None else Path(reference_directory)
    old, new = snapshot(plotfiles(original)[0]), snapshot(plotfiles(directory)[-1])
    assert old["time"] == 0 and abs(new["time"] / 2.56e-9 - 1) < 1e-12
    assert np.array_equal(old["ids"], new["ids"])
    assert np.array_equal(old["weight"], new["weight"])
    before = np.loadtxt(original / "native_inventory.txt", dtype=np.longdouble)[0]
    after = np.loadtxt(directory / "native_inventory.txt", dtype=np.longdouble)[1]
    initial_l, final_l = angular_inventory(old), angular_inventory(new)
    delta_l = final_l - initial_l
    angular_scale = before[2] * np.longdouble("0.001") / C
    angular_bound = 1e-10 * angular_scale + 128 * np.finfo(float).eps * np.sum(
        abs(initial_l)
    )
    mass = MP * old["weight"]
    g0 = np.sqrt(1 + np.sum(old["u"] ** 2, axis=1) / C**2)
    g1 = np.sqrt(1 + np.sum(new["u"] ** 2, axis=1) / C**2)
    work = np.sum(
        mass * np.sum((new["u"] - old["u"]) * (new["u"] + old["u"]), axis=1) / (g0 + g1)
    )
    carry_work = np.sum(mass * (new["carry_work"] - old["carry_work"]))
    radiation_change = np.sum(
        new["radiation_diffusion_energy"] - old["radiation_diffusion_energy"]
    )
    energy_error = work + carry_work + radiation_change + after[1] - before[1]
    energy_bound = 1e-10 * before[2] + 128 * np.finfo(float).eps * np.sum(abs(before))
    turn = (new["theta"] - old["theta"] + np.pi) % (2 * np.pi) - np.pi
    displacement = new["r"] - old["r"]
    result = {
        "time_s": new["time"],
        "angular_change_kg_m2_s": [float(x) for x in delta_l],
        "angular_error_kg_m2_s": float(np.sum(delta_l)),
        "angular_bound_kg_m2_s": float(angular_bound),
        "mean_rotation_rad": float(np.mean(turn)),
        "maximum_radial_displacement_m": float(np.max(abs(displacement))),
        "kinetic_change_J": float(work),
        "radiation_change_J": float(radiation_change),
        "electron_change_J": float(after[1] - before[1]),
        "energy_error_J": float(energy_error),
        "energy_bound_J": float(energy_bound),
        "gates_passed": False,
    }
    output = directory / "rz_moving_rotation.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    assert np.mean(turn) > 0.05, result
    assert np.max(abs(displacement)) > 0.25e-3 / old["rho"].shape[0], result
    assert delta_l[0] > 1e-8 * angular_scale, result
    assert abs(np.sum(delta_l)) <= angular_bound, result
    assert abs(energy_error) <= energy_bound, result
    diagnostic = directory / "diags" / "radiation_momentum.txt"
    columns = {
        name: int(index)
        for index, name in re.findall(
            r"\[(\d+)\]([^\s(]+)", diagnostic.read_text().splitlines()[0]
        )
    }
    history = np.atleast_2d(np.loadtxt(diagnostic, dtype=np.longdouble))
    row = history[-1]
    for value, account in zip(
        final_l, ("radiation", "represented_material", "pending_material")
    ):
        assert abs(row[columns[f"angular_{account}_z"]] - value) <= (
            1e-12 * abs(value) + 128 * np.finfo(float).eps * angular_scale
        ), result
    if reference_directory is not None:
        reference = snapshot(plotfiles(original)[-1])
        assert np.array_equal(new["ids"], reference["ids"])
        for name, scale in (("r", 1e-3), ("z", 1e-3), ("u", 1e5), ("theta", 1)):
            assert np.max(abs(new[name] - reference[name])) <= 1e-11 * scale
        assert np.max(abs(final_l - angular_inventory(reference))) <= angular_bound
        result["restart_matches"] = True
    result["gates_passed"] = True
    output.write_text(json.dumps(result, indent=2) + "\n")
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    r = (np.arange(old["rho"].shape[0]) + 0.5) / old["rho"].shape[0]
    for s, label in ((old, "initial"), (new, "final")):
        axes[0].plot(r, s["rho"].mean(axis=1) / old["rho"].mean(), label=label)
    axes[0].set(xlabel="r (mm)", ylabel="density / initial mean")
    for sign, account, label in (
        (1, "radiation", "radiation gain"),
        (-1, "represented_material", "represented-ion loss"),
    ):
        values = history[:, columns[f"angular_{account}_z"]]
        axes[1].plot(
            history[:, 1] * 1e9,
            sign * (values - initial_l[0 if sign == 1 else 1]),
            label=label,
        )
    total = sum(
        history[:, columns[f"angular_{account}_z"]]
        for account in ("radiation", "represented_material", "pending_material")
    )
    axes[1].set(xlabel="time (ns)", ylabel="angular transfer (kg m²/s)")
    axes[2].plot(history[:, 1] * 1e9, (total - initial_l.sum()) / angular_bound)
    axes[2].axhline(1, color="gray", linestyle="--")
    axes[2].axhline(-1, color="gray", linestyle="--")
    axes[2].set(xlabel="time (ns)", ylabel="angular error / allowed bound")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    figure.savefig(directory / "rz_moving_rotation.png", dpi=160)
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference-directory", type=Path)
    args = parser.parse_args()
    check(args.directory, args.reference_directory)
