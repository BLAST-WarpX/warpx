#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Space/time interface gates, without fitting or subtracting material controls."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analysis_moving_moment_interface import caloric, check, load
from analysis_moving_moment_pulse import plotfiles


def fields(row):
    # Conservative cell restriction: use cell radiation densities and arithmetic
    # cell-corner native charge/caloric densities, not T times averaged density.
    return {
        "radiation_energy_density": row["radiation_diffusion_energy"] / row["dx"],
        "radiation_qz_density": row["radiation_moment_qz"] / row["dx"],
        "charge_density": (row["rho_nodes"] + np.roll(row["rho_nodes"], -1)) / 2,
        "electron_caloric_density": (caloric(row) + np.roll(caloric(row), -1)) / 2,
    }


def analyze(directories, output, temporal=False):
    output.mkdir(parents=True, exist_ok=True)
    contracts = []
    for directory in directories:
        contract = {}
        for line in (directory / "warpx_used_inputs").read_text().splitlines():
            key, separator, value = line.partition("=")
            key = key.strip()
            if separator and (
                key.startswith(
                    (
                        "algo.",
                        "hybrid_pic_model.",
                        "ions.",
                        "radiation_transport.",
                        "my_constants.",
                        "geometry.",
                        "boundary.",
                        "warpx.",
                        "interpolation.",
                    )
                )
                or key in ("warpx.const_dt", "max_step")
            ):
                contract[key] = value.split("#", 1)[0].strip()
        assert "algo.particle_shape" in contract
        assert contract["warpx.grid_type"] == "collocated"
        # Collocated gathering defaults to the full shape. Earlier archived
        # runs used that default rather than spelling out the valid parameter.
        contract.setdefault("interpolation.galerkin_scheme", "0")
        split_boxes = contract.get("warpx.split_high_density_boxes", "false")
        if split_boxes in ("0", "false"):
            contract["warpx.split_high_density_boxes"] = "false"
            # MPI may record these defaults even though splitting is disabled.
            contract.pop("warpx.split_high_density_boxes_min_box_size", None)
            contract.pop("warpx.split_high_density_boxes_threshold", None)
        if temporal:
            del contract["warpx.const_dt"]
            del contract["max_step"]
        contracts.append(contract)
    assert contracts[0] == contracts[1] == contracts[2], (
        "Mesh refinement must preserve particle shape, sampling, physics and timestep."
    )
    reports = [check(path, require_laminar=True) for path in directories]
    states = [load(plotfiles(path)[-1]) for path in directories]
    if temporal:
        assert states[0]["n"] == states[1]["n"] == states[2]["n"]
        assert states[1]["step"] == 2 * states[0]["step"]
        assert states[2]["step"] == 2 * states[1]["step"]
    else:
        assert states[1]["n"] == 2 * states[0]["n"]
        assert states[2]["n"] == 2 * states[1]["n"]
    length = states[0]["n"] * states[0]["dx"]
    for row in states:
        assert abs(row["time"] / states[0]["time"] - 1) < 1e-12
        assert abs(row["n"] * row["dx"] / length - 1) < 1e-12
    values = [fields(row) for row in states]
    errors = []
    for coarse, fine in zip(values[:-1], values[1:], strict=True):
        restricted = {
            name: value if temporal else value.reshape(-1, 2).mean(axis=1)
            for name, value in fine.items()
        }
        errors.append(
            {
                name: float(
                    np.sum(abs(coarse[name] - restricted[name]))
                    / np.sum(abs(restricted[name]))
                )
                for name in coarse
            }
        )
    ratios = {name: errors[0][name] / errors[1][name] for name in errors[0]}
    kinetic = [row["history"][-1]["kinetic_change_J"] for row in reports]
    kinetic_ratio = abs((kinetic[0] - kinetic[1]) / (kinetic[1] - kinetic[2]))
    result = {
        "refinement_kind": "temporal" if temporal else "spatial",
        "directories": [str(path.resolve()) for path in directories],
        "cells": [row["n"] for row in states],
        "steps": [row["step"] for row in states],
        "relative_l1_self_differences": errors,
        "refinement_ratios": ratios,
        "minimum_required_ratio": 1.5,
        "kinetic_changes_J": kinetic,
        "kinetic_work_refinement_ratio": kinetic_ratio,
    }
    (output / "refinement.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, name in zip(axes.flat, values[0], strict=True):
        for row, data in zip(states, values, strict=True):
            x = (np.arange(row["n"]) + 0.5) * float(row["dx"]) * 1e3
            label = f"steps={row['step']}" if temporal else f"N={row['n']}"
            axis.plot(x, data[name], label=label)
        axis.set_title(name.replace("_", " ") + " [SI]")
        axis.set_xlabel("Position [mm]")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.savefig(output / "profiles.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result, indent=2))
    # A first-order method should approach ratio two. This lower gate requires
    # clear reduction in every inventory field, not just a prettier energy plot.
    assert all(value >= 1.5 for value in ratios.values()), ratios
    if temporal:
        assert kinetic_ratio >= 1.5, kinetic_ratio
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs=3, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--temporal", action="store_true")
    args = parser.parse_args()
    analyze(args.directories, args.output, args.temporal)
