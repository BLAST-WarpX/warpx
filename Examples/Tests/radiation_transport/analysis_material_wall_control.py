#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Native reflecting material control; radiation is disabled, no FLASH closure."""

import argparse
import json
from pathlib import Path

import numpy as np
import yt
from analysis_moving_moment_interface import caloric, nodes
from analysis_moving_moment_pulse import KB, MP, QE, C, plotfiles

yt.set_log_level(50)


def load(path):
    ds = yt.load(str(path))
    assert ds.dimensionality == 1
    cells = int(ds.domain_dimensions[0])
    dx = np.longdouble(ds.domain_width[0].v) / cells
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
        "ids": np.stack((particle("id"), particle("cpu")), axis=1),
        "position": particle("position_x"),
        "mass": particle("weight") * MP,
        "u": np.stack([particle("momentum_" + axis) / MP for axis in "xyz"], axis=1),
        "rho_nodes": nodes(path / "raw_fields/Level_0/rho_fp_H", cells, periodic=False),
        "temperature_nodes": nodes(
            path.parent
            / f"chk{step:06d}/Level_0/hybrid_electron_temperature_fp[level=0]_H",
            cells,
            periodic=False,
        ),
    }
    volumes = np.full(cells + 1, dx, dtype=np.longdouble)
    volumes[[0, -1]] *= 0.5
    row["charge_mass"] = np.sum(row["rho_nodes"] * volumes) * MP / QE
    row["caloric"] = np.sum(caloric(row) * volumes)
    for value in row.values():
        assert np.isfinite(value).all()
    assert np.min(row["temperature_nodes"]) > 0
    return row


def check(directory, initial_only=False, reference=None, plot=False):
    initial = None if reference is None else load(plotfiles(reference)[0])
    history = []
    if initial is not None:
        assert initial["step"] == 0
        gamma0 = np.sqrt(1 + np.sum(initial["u"] ** 2, axis=1) / C**2)
        kinetic0 = np.sum(
            initial["mass"] * np.sum(initial["u"] ** 2, axis=1) / (gamma0 + 1)
        )
        energy0 = kinetic0 + initial["caloric"]
    for path in plotfiles(directory):
        current = load(path)
        if initial is None:
            initial = current
            assert initial["step"] == 0
            gamma0 = np.sqrt(1 + np.sum(initial["u"] ** 2, axis=1) / C**2)
            kinetic0 = np.sum(
                initial["mass"] * np.sum(initial["u"] ** 2, axis=1) / (gamma0 + 1)
            )
            energy0 = kinetic0 + initial["caloric"]
        assert np.array_equal(initial["ids"], current["ids"])
        assert np.array_equal(initial["mass"], current["mass"])
        gamma = np.sqrt(1 + np.sum(current["u"] ** 2, axis=1) / C**2)
        kinetic_change = np.sum(
            initial["mass"]
            * np.sum(
                (current["u"] - initial["u"]) * (current["u"] + initial["u"]), axis=1
            )
            / (gamma + gamma0)
        )
        heat_change = current["caloric"] - initial["caloric"]
        mass = np.sum(current["mass"])
        history.append(
            {
                "step": current["step"],
                "time_s": current["time"],
                "particle_mass": float(mass),
                "nodal_charge_mass": float(current["charge_mass"]),
                "mass_relative_error": float(abs(current["charge_mass"] - mass) / mass),
                "kinetic_change": float(kinetic_change),
                "caloric_change": float(heat_change),
                "energy_relative_error": float(
                    abs(kinetic_change + heat_change) / energy0
                ),
                "negative_longitudinal_momentum_fraction": float(
                    np.mean(current["u"][:, 2] < 0)
                ),
            }
        )
    assert initial is not None
    report = {
        "initial_only": initial_only,
        "reference_requested": reference is not None,
        "changed_rank_restart_matches": False,
        "all_gates_passed": False,
        "initial_total_energy": float(energy0),
        "history": history,
    }
    try:
        if reference is not None:
            baseline = load(plotfiles(reference)[-1])
            report["reference_max_absolute_differences"] = {
                name: float(np.max(abs(current[name] - baseline[name])))
                for name in ("position", "u", "rho_nodes", "temperature_nodes")
            }
            assert current["step"] == baseline["step"]
            assert np.array_equal(current["ids"], baseline["ids"])
            assert np.array_equal(current["mass"], baseline["mass"])
            assert (
                np.max(abs(current["position"] - baseline["position"])) < 1e-11 * 0.001
            )
            assert np.max(abs(current["u"] - baseline["u"])) < 1e-12 * 9e4
            for name in ("rho_nodes", "temperature_nodes"):
                assert np.max(abs(current[name] - baseline[name])) < 1e-10 * np.max(
                    abs(baseline[name])
                )
            report["changed_rank_restart_matches"] = True
        print(json.dumps(history[-1], indent=2))
        assert max(row["mass_relative_error"] for row in history) < 1e-12, (
            "Physical nodal mass mismatch"
        )
        assert max(row["energy_relative_error"] for row in history) < 1e-8, (
            "Native wall energy mismatch"
        )
        if initial_only:
            assert all(row["step"] == 0 for row in history)
        else:
            assert history[-1]["time_s"] >= 3.2e-9 * (1 - 1e-12)
            assert history[-1]["negative_longitudinal_momentum_fraction"] > 0.01
        report["all_gates_passed"] = True
    finally:
        # Preserve quantitative failure evidence without suppressing any assertion.
        (directory / "material_wall_metrics.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        times = [row["time_s"] * 1e9 for row in history]
        for name, label in (
            ("kinetic_change", "Ion kinetic"),
            ("caloric_change", "Electron"),
        ):
            axes[0].plot(times, [row[name] / 1e6 for row in history], "o-", label=label)
        axes[0].set(xlabel="Time [ns]", ylabel="Energy change [MJ, unit area]")
        axes[0].legend()
        z = np.linspace(0, 1, len(current["temperature_nodes"]))
        for row, label in ((initial, "Initial"), (current, "Final")):
            axes[1].plot(z, row["temperature_nodes"] * KB / QE, label=label)
            axes[2].scatter(
                row["position"] * 1e3, row["u"][:, 2] / 1e3, s=4, label=label
            )
        axes[1].set(xlabel="z [mm]", ylabel="Native electron temperature [eV]")
        axes[2].set(xlabel="z [mm]", ylabel="Ion proper velocity [km/s]")
        axes[1].legend()
        axes[2].legend()
        fig.suptitle("Native reflecting material wall — radiation disabled")
        fig.savefig(directory / "material_wall_control.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--initial-only", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    check(args.directory, args.initial_only, args.reference, args.plot)
