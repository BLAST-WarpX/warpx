#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Native photon/P1 round trip: actual inventories and independent finite-mass BE."""

import argparse
import json
from pathlib import Path

import numpy as np
import yt
from scipy.optimize import brentq

C = np.longdouble("299792458")
MP = np.longdouble("1.67262192595e-27")
KB = np.longdouble("1.380649e-23")
QE = np.longdouble("1.602176634e-19")
yt.funcs.mylog.setLevel(40)


def load(path):
    ds = yt.load(str(path))
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    data = ds.all_data()
    order = np.lexsort((data["ions", "particle_cpu"].v, data["ions", "particle_id"].v))

    def ion(name):
        return np.asarray(data["ions", "particle_" + name].v, dtype=np.longdouble)[
            order
        ]

    def field(name):
        return np.asarray(grid["boxlib", name].v, dtype=np.longdouble).ravel()

    masses = MP * ion("weight")
    u = np.stack([ion("momentum_" + a) / MP for a in "xyz"], axis=1)
    carry = np.stack(
        [ion("radiation_impulse_diffusion_0_u" + a) for a in "xyz"], axis=1
    )
    photon_q = np.zeros(3, dtype=np.longdouble)
    photon_e = np.longdouble(0)
    nphotons = 0
    if ("photons", "particle_weight") in ds.field_list:
        weights = np.asarray(data["photons", "particle_weight"].v, dtype=np.longdouble)
        momentum = np.stack(
            [
                np.asarray(
                    data["photons", "particle_momentum_" + a].v, dtype=np.longdouble
                )
                for a in "xyz"
            ],
            axis=1,
        )
        nphotons = len(weights)
        photon_e = np.sum(weights * np.sqrt(np.sum(momentum**2, axis=1))) * C
        photon_q = np.sum(weights[:, None] * momentum, axis=0) * C
    field_e = np.sum(field("radiation_diffusion_energy"))
    field_q = np.array([np.sum(field("radiation_flux_relaxed_q" + a)) for a in "xyz"])
    volume = np.prod(np.asarray(ds.domain_width.v, dtype=np.longdouble))
    internal = np.mean(field("rho") / QE * 1.5 * KB * field("Te")) * volume
    return dict(
        step=int(path.name[5:]),
        time=float(ds.current_time.v),
        ids=np.stack((ion("id"), ion("cpu")), axis=1),
        mass=masses,
        u=u,
        carry=carry,
        carry_work=ion("radiation_impulse_diffusion_0_work"),
        photon_e=photon_e,
        field_e=field_e,
        energy=photon_e + field_e,
        q=photon_q + field_q,
        internal=internal,
        nphotons=nphotons,
        position=ion("position_x"),
    )


def reference(initial, steps):
    energy = initial["energy"]
    q = initial["q"][2]
    mass = initial["mass"].sum()
    u = np.sum(initial["mass"] * initial["u"][:, 2]) / mass
    history = {0: (energy, q, u)}
    for step in range(1, steps + 1):
        if step <= 20:
            beta = u / np.sqrt(1 + (u / C) ** 2) / C
            x = 200 * C * np.longdouble("1e-11")

            def state(enew):
                impulse = x / (1 + x) * (q - 4 * beta * enew / 3) / C
                unew = u + impulse / mass
                work = (
                    mass
                    * (unew - u)
                    * (unew + u)
                    / (np.sqrt(1 + (unew / C) ** 2) + np.sqrt(1 + (u / C) ** 2))
                )
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
        history[step] = energy, q, u
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
    assert initial["nphotons"] == 128
    assert abs(initial["energy"] / 1e4 - 1) < 1e-12
    expected = reference(initial, 40)
    records = []
    mass = initial["mass"]
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
        u_mean = np.sum(mass * row["u"][:, 2]) / mass.sum()
        assert abs(row["energy"] - e_ref) < 2e-10 * initial["energy"]
        assert abs(row["q"][2] - q_ref) < 2e-10 * initial["energy"]
        assert abs(u_mean - u_ref) < 2e-9 * C * initial["energy"] / (mass.sum() * C * C)
        if 0 < row["step"] <= 20:
            assert row["nphotons"] == 0 and row["field_e"] > 0
        if row["step"] >= 25:
            assert row["nphotons"] == 256 and row["field_e"] == 0
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
    assert records[-1]["ion_kinetic_gain_J"] > 0.5
    if reference_directory:
        baseline = prior[-1]
        final = rows[-1]
        assert abs(final["energy"] - baseline["energy"]) < 1e-11 * initial["energy"]
        assert np.max(abs(final["q"] - baseline["q"])) < 1e-11 * initial["energy"]
        assert np.max(abs(final["u"] - baseline["u"])) < 1e-8
    result = dict(
        model="gray scattering flux-relaxed P1; not general FLD", history=records
    )
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
