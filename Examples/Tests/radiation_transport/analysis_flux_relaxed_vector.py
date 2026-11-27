#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Native gray packet 3D vector round trip: 3D inventories and vector BE oracle."""

import argparse
import json
from pathlib import Path

import numpy as np
from analysis_flux_relaxed_roundtrip import MP, C, load
from scipy.optimize import brentq


def reference(initial, steps):
    energy = initial["energy"]
    q = np.array(initial["q"], dtype=np.longdouble)
    mass = initial["mass"].sum()
    u = np.sum(initial["mass"][:, None] * initial["u"], axis=0) / mass
    history = {0: (energy, q.copy(), u.copy())}
    dt = np.longdouble("1e-11")
    for step in range(1, steps + 1):
        if step <= 20:
            gamma = np.sqrt(1 + np.sum(u**2) / C**2)
            beta = u / (gamma * C)
            x = 200 * C * dt

            def state(enew):
                impulse = x / (1 + x) * (q - 4 * beta * enew / 3) / C
                unew = u + impulse / mass
                gamma_new = np.sqrt(1 + np.sum(unew**2) / C**2)
                work = mass * np.dot(unew - u, unew + u) / (gamma_new + gamma)
                return impulse, unew, work

            def residual(enew):
                return float(enew - energy + state(np.longdouble(enew))[2])

            enew = np.longdouble(
                brentq(
                    residual,
                    float(energy * 0.99),
                    float(energy * 1.01),
                    xtol=1e-11,
                    rtol=1e-14,
                )
            )
            impulse, u, _ = state(enew)
            energy, q = enew, q - C * impulse
        history[step] = (energy, q.copy(), u.copy())
    return history


def check(directory, reference_directory=None):
    def files(root):
        return sorted(
            p for p in (root / "diags").glob("diag1*") if p.name[5:].isdigit()
        )

    paths = files(directory)
    rows = [load(p) for p in paths]
    if reference_directory:
        prior = [load(p) for p in files(reference_directory)]
        rows = [r for r in prior if r["step"] < rows[0]["step"]] + rows
    initial = rows[0]
    assert initial["step"] == 0 and rows[-1]["step"] == 40
    assert initial["nphotons"] == 4096
    assert abs(initial["energy"] / 1e4 - 1) < 1e-12
    expected = reference(initial, 40)
    records = []
    mass = initial["mass"]
    total_mass = mass.sum()
    assert abs(total_mass / (MP * np.longdouble("1e20")) - 1) < 1e-12
    assert (
        np.max(abs(initial["q"] / initial["energy"] - np.array([0, 0.6, 0.8]))) < 1e-12
    )
    gamma0 = np.sqrt(1 + np.sum(initial["u"] ** 2, axis=1) / C**2)
    for row in rows:
        assert np.array_equal(row["ids"], initial["ids"])
        assert np.array_equal(row["mass"], mass)
        du = row["u"] - initial["u"]
        gamma = np.sqrt(1 + np.sum(row["u"] ** 2, axis=1) / C**2)
        kinetic = np.sum(
            mass * np.sum(du * (row["u"] + initial["u"]), axis=1) / (gamma + gamma0)
        )
        carry_work = np.sum(mass * (row["carry_work"] - initial["carry_work"]))
        energy_error = (
            row["energy"]
            - initial["energy"]
            + kinetic
            + carry_work
            + row["internal"]
            - initial["internal"]
        )
        momentum_error = (row["q"] - initial["q"]) / C + np.sum(
            mass[:, None] * (du + row["carry"] - initial["carry"]), axis=0
        )
        assert abs(energy_error) < 1e-10 * initial["energy"]
        assert np.max(abs(momentum_error)) < 1e-10 * initial["energy"] / C
        assert abs(row["internal"] - initial["internal"]) < 1e-10 * initial["energy"]
        e_ref, q_ref, u_ref = expected[row["step"]]
        u_mean = np.sum(mass[:, None] * row["u"], axis=0) / total_mass
        assert abs(row["energy"] - e_ref) < 2e-10 * initial["energy"]
        assert np.max(abs(row["q"] - q_ref)) < 2e-10 * initial["energy"]
        assert np.max(abs(u_mean - u_ref)) < 2e-9 * initial["energy"] / (total_mass * C)
        if 0 < row["step"] <= 20:
            assert row["nphotons"] == 0 and row["field_e"] > 0
        if row["step"] >= 25:
            assert row["nphotons"] == 4096 and row["field_e"] == 0
        records.append(
            dict(
                step=row["step"],
                photons=row["nphotons"],
                field_energy_J=float(row["field_e"]),
                photon_energy_J=float(row["photon_e"]),
                ion_kinetic_gain_J=float(kinetic),
                raw_energy_error_J=float(energy_error),
                raw_momentum_error_kgms=list(map(float, momentum_error)),
            )
        )
    du_final = (
        np.sum(mass[:, None] * (rows[-1]["u"] - initial["u"]), axis=0) / total_mass
    )
    assert du_final[0] < 0
    assert du_final[1] > 0
    assert du_final[2] > 0
    assert records[-1]["ion_kinetic_gain_J"] > 0.001
    if reference_directory:
        baseline = prior[-1]
        final = rows[-1]
        assert abs(final["energy"] - baseline["energy"]) < 1e-11 * initial["energy"]
        assert np.max(abs(final["q"] - baseline["q"])) < 1e-11 * initial["energy"]
        assert np.max(abs(final["u"] - baseline["u"])) < 1e-8
    result = dict(model="gray scattering flux-relaxed P1 3D vector", history=records)
    (directory / "flux_relaxed_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(records[-1], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    check(args.directory, args.reference)
