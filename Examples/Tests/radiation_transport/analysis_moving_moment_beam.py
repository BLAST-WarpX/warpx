#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Independent homogeneous four-force ODE and actual moving PIC inventories."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analysis_moving_moment_interface import caloric, load
from analysis_moving_moment_pulse import MP, C, plotfiles
from scipy.integrate import solve_ivp


def reference(tau):
    epsilon = 0.005
    x0 = 0.002 / np.sqrt(1 - 0.002**2)
    gamma0 = np.sqrt(1 + x0**2)

    def state(q):
        x = x0 + epsilon * (1 - q)
        gamma = np.sqrt(1 + x**2)
        # Stable gamma difference, independent of the WarpX source kernels.
        e = 1 - (x - x0) * (x + x0) / (epsilon * (gamma + gamma0))
        return x, gamma, e

    def rhs(_, values):
        q = values[0]
        x, gamma, e = state(q)
        beta = x / gamma
        f = q / e
        chi = (3 + 4 * f**2) / (5 + 2 * np.sqrt(4 - 3 * f**2))
        return [-(gamma**3) * ((1 + beta**2) * q - beta * (1 + chi) * e), beta]

    solution = solve_ivp(
        rhs,
        (0, float(tau[-1])),
        [1, 0],
        t_eval=tau,
        method="DOP853",
        rtol=1e-13,
        atol=1e-15,
    )
    assert solution.success
    q = solution.y[0]
    x, _, e = state(q)
    return {"q": q, "e": e, "x": x, "travel_m": solution.y[1] / 1e4}


def check(directory, check_momentum_diagnostic=False, restart_reference=None):
    files = plotfiles(directory)
    reference_files = None
    if restart_reference is not None:
        reference_files = plotfiles(restart_reference)
        first_step = int(files[0].name[5:])
        files = [
            path for path in reference_files if int(path.name[5:]) < first_step
        ] + files
    rows = [load(path) for path in files]
    initial = rows[0]
    assert initial["step"] == 0 and len(rows) >= 21
    tau = np.asarray([float(C) * 1e4 * row["time"] for row in rows])
    assert abs(tau[-1] / 10 - 1) < 1e-12
    expected = reference(tau)
    mass = initial["weight"] * MP
    energy0 = np.sum(initial["radiation_diffusion_energy"])
    momentum_diagnostic = None
    if check_momentum_diagnostic:
        momentum_diagnostic = np.loadtxt(directory / "diags/radiation_momentum.txt")
        if restart_reference is not None:
            prior = np.loadtxt(restart_reference / "diags/radiation_momentum.txt")
            momentum_diagnostic = np.concatenate(
                (prior[prior[:, 0] < momentum_diagnostic[0, 0]], momentum_diagnostic)
            )
        assert momentum_diagnostic.shape[1] == 29
        assert np.all(np.isfinite(momentum_diagnostic))
        inventory = (
            momentum_diagnostic[:, 26:29]
            + momentum_diagnostic[:, 5:8]
            + momentum_diagnostic[:, 11:14]
            + momentum_diagnostic[:, 17:20]
            + momentum_diagnostic[:, 20:23]
            + momentum_diagnostic[:, 23:26]
        )
        assert np.max(abs(inventory - inventory[0])) < 1e-10 * energy0 / C
    assert abs(energy0 / (np.sum(mass) * C**2) / 0.005 - 1) < 1e-12
    gamma0 = np.sqrt(1 + np.sum(initial["u"] ** 2, axis=1) / C**2)
    assert np.max(abs(initial["u"][:, 2] / gamma0 / C - 0.002)) < 1e-14
    length = initial["n"] * initial["dx"]
    initial_internal = caloric(initial)
    travel = np.zeros_like(initial["position"])
    previous_position = initial["position"]
    history = []
    for row in rows:
        assert np.array_equal(row["ids"], initial["ids"])
        assert np.array_equal(row["weight"], initial["weight"])
        if momentum_diagnostic is not None:
            record = momentum_diagnostic[momentum_diagnostic[:, 0] == row["step"]]
            assert record.shape[0] == 1
            field_momentum = np.array(
                [np.sum(row["radiation_moment_q" + axis]) / C for axis in "xyz"]
            )
            assert np.max(abs(record[0, 26:29] - field_momentum)) < 1e-12 * energy0 / C
        displacement = row["position"] - previous_position
        travel += (displacement + length / 2) % length - length / 2
        previous_position = row["position"]
        gamma = np.sqrt(1 + np.sum(row["u"] ** 2, axis=1) / C**2)
        du = row["u"] - initial["u"]
        kinetic = np.sum(
            mass * np.sum(du * (row["u"] + initial["u"]), axis=1) / (gamma0 + gamma)
        )
        internal = np.sum(caloric(row) - initial_internal) * initial["dx"]
        carry = np.sum(mass * (row["carry_work"] - initial["carry_work"]))
        radiation = np.sum(
            row["radiation_diffusion_energy"] - initial["radiation_diffusion_energy"]
        )
        assert abs(radiation + kinetic + internal + carry) < 1e-10 * energy0
        momentum_errors = []
        for d, axis in enumerate("xyz"):
            momentum = (
                np.sum(
                    row["radiation_moment_q" + axis]
                    - initial["radiation_moment_q" + axis]
                )
                / C
            )
            momentum += np.sum(
                mass * (du[:, d] + row["carry"][:, d] - initial["carry"][:, d])
            )
            momentum_errors.append(float(momentum))
            assert abs(momentum) < 1e-10 * energy0 / C + 128 * np.finfo(
                float
            ).eps * np.sum(mass * abs(initial["u"][:, d]))
        assert (
            np.max(abs(row["temperature_nodes"] / initial["temperature_nodes"] - 1))
            < 1e-8
        )
        assert abs(internal) < 1e-8 * np.sum(initial_internal) * initial["dx"]
        assert np.ptp(row["rho_nodes"]) < 1e-10 * np.mean(row["rho_nodes"])
        assert np.ptp(row["radiation_diffusion_energy"]) < 1e-10 * np.mean(
            row["radiation_diffusion_energy"]
        )
        assert np.ptp(row["u"][:, 2]) < 1e-10 * np.mean(row["u"][:, 2])
        history.append(
            {
                "tau": float(C * 1e4 * row["time"]),
                "e": float(np.sum(row["radiation_diffusion_energy"]) / energy0),
                "q": float(np.sum(row["radiation_moment_qz"]) / energy0),
                "x": float(np.sum(mass * row["u"][:, 2]) / np.sum(mass) / C),
                "travel_m": float(np.sum(mass * travel) / np.sum(mass)),
                "kinetic_gain_over_radiation": float(kinetic / energy0),
                "raw_energy_error_over_radiation": float(
                    (radiation + kinetic + internal + carry) / energy0
                ),
                "raw_momentum_errors": momentum_errors,
                "electron_caloric_change_J": float(internal),
                "maximum_temperature_relative_change": float(
                    np.max(
                        abs(row["temperature_nodes"] / initial["temperature_nodes"] - 1)
                    )
                ),
            }
        )
    assert history[-1]["kinetic_gain_over_radiation"] > 1e-3
    assert np.min(travel) > 0.5 * length
    actual = {name: np.asarray([row[name] for row in history]) for name in expected}
    scales = {
        "e": 1 - expected["e"][-1],
        "q": 1,
        "x": expected["x"][-1] - expected["x"][0],
        "travel_m": float(length),
    }
    errors = {
        name: float(np.max(abs(actual[name] - expected[name])) / scales[name])
        for name in expected
    }
    ledger = np.loadtxt(directory / "diags/radiation_energy.txt")
    assert (
        np.max(abs(ledger[:, 2] + ledger[:, 6] + ledger[:, 16] - energy0))
        < 1e-10 * energy0
    )
    report = {
        "steps": rows[-1]["step"],
        "scaled_max_errors": errors,
        "history": history,
    }
    if reference_files is not None:
        baseline = load(reference_files[-1])
        current = rows[-1]
        assert current["step"] == baseline["step"]
        delta = (
            current["position"] - baseline["position"] + length / 2
        ) % length - length / 2
        assert np.max(abs(delta)) < 1e-11 * length
        assert np.max(abs(current["u"] - baseline["u"])) < 1e-12 * np.max(
            abs(baseline["u"])
        )
        for name in (
            "radiation_diffusion_energy",
            "radiation_moment_qx",
            "radiation_moment_qy",
            "radiation_moment_qz",
            "rho_nodes",
            "temperature_nodes",
        ):
            assert np.max(abs(current[name] - baseline[name])) <= 1e-10 * np.max(
                abs(baseline[name])
            ), name
        report["changed_rank_restart_matches"] = True
    (directory / "beam_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    curve_tau = np.linspace(0, 10, 401)
    curve = reference(curve_tau)
    for axis, name in zip(axes, ("q", "x", "travel_m"), strict=True):
        axis.plot(curve_tau, curve[name], "k-", label="Independent ODE")
        axis.plot(tau, actual[name], "o", label="Native PIC")
        axis.set_xlabel("Optical time c kappa t")
        axis.set_ylabel(
            {
                "q": "Radiation flux F/(c E0)",
                "x": "Ion proper velocity u/c",
                "travel_m": "Particle travel [m]",
            }[name]
        )
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.savefig(directory / "beam_reference.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    assert errors["q"] < 1e-2 and errors["x"] < 1e-2
    assert errors["e"] < 2e-2 and errors["travel_m"] < 1e-2
    # At late time the nonzero comoving flux is essential: a stationary-source
    # approximation must not pass solely because transient errors allow 1%.
    assert abs(actual["q"][-1] - expected["q"][-1]) < 1e-4
    assert abs(actual["x"][-1] - expected["x"][-1]) < 1e-4 * scales["x"]
    assert abs(actual["e"][-1] - expected["e"][-1]) < 1e-3 * scales["e"]
    return report


def check_temporal(directories, output, check_momentum_diagnostic=False):
    reports = [check(directory, check_momentum_diagnostic) for directory in directories]
    assert reports[1]["steps"] == 2 * reports[0]["steps"]
    assert reports[2]["steps"] == 2 * reports[1]["steps"]
    for report in reports[1:]:
        assert np.allclose(
            [row["tau"] for row in report["history"]],
            [row["tau"] for row in reports[0]["history"]],
            rtol=1e-12,
            atol=1e-14,
        )
    ratios = [
        {
            name: coarse["scaled_max_errors"][name] / fine["scaled_max_errors"][name]
            for name in coarse["scaled_max_errors"]
        }
        for coarse, fine in zip(reports[:-1], reports[1:], strict=True)
    ]
    result = {
        "steps": [row["steps"] for row in reports],
        "errors": [row["scaled_max_errors"] for row in reports],
        "refinement_ratios": ratios,
        "minimum_required_ratio": 1.8,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "beam_refinement.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert all(value >= 1.8 for row in ratios for value in row.values()), ratios
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-momentum-diagnostic", action="store_true")
    parser.add_argument("--restart-reference", type=Path)
    args = parser.parse_args()
    if len(args.directories) == 1:
        check(
            args.directories[0], args.check_momentum_diagnostic, args.restart_reference
        )
    elif len(args.directories) == 3 and args.output is not None:
        assert args.restart_reference is None
        check_temporal(args.directories, args.output, args.check_momentum_diagnostic)
    else:
        parser.error("Provide one case, or three temporal cases with --output.")
