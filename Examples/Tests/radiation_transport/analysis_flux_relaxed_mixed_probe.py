#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Prospective mixed K/D inventory gate; current cold-slab runtime rejects.

This unregistered probe is not a qualified passing regression. Incomplete
histories must fail before a success receipt can be emitted.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yt
from analysis_flux_relaxed_roundtrip import C, load

yt.funcs.mylog.setLevel(40)

K_ZMIN = np.longdouble("0.375")
K_ZMAX = np.longdouble("0.5")
D_ZMIN = np.longdouble("0.5")
D_ZMAX = np.longdouble("0.75")
EXPECTED_DIRECTION = np.array([0.3, 0.4, np.sqrt(0.75)], dtype=np.longdouble)


def plotfiles(directory):
    return sorted(
        path for path in (directory / "diags").glob("diag1*") if path.name[5:].isdigit()
    )


def photons(path):
    data = yt.load(str(path)).all_data()
    if ("photons", "particle_weight") not in data.ds.field_list:
        return dict(
            ids=np.array([], dtype=np.int64),
            cpu=np.array([], dtype=np.int64),
            weight=np.array([], dtype=np.longdouble),
            z=np.array([], dtype=np.longdouble),
            momentum=np.empty((0, 3), dtype=np.longdouble),
            mass=np.array([], dtype=np.longdouble),
        )

    ids = np.asarray(data["photons", "particle_id"].v, dtype=np.int64)
    cpu = np.asarray(data["photons", "particle_cpu"].v, dtype=np.int64)
    order = np.lexsort((cpu, ids))
    momentum = np.stack(
        [
            np.asarray(
                data["photons", "particle_momentum_" + axis].v, dtype=np.longdouble
            )
            for axis in "xyz"
        ],
        axis=1,
    )
    if ("photons", "particle_mass") in data.ds.field_list:
        mass = np.asarray(data["photons", "particle_mass"].v, dtype=np.longdouble)
    else:
        mass = np.zeros(len(ids), dtype=np.longdouble)
    return dict(
        ids=ids[order],
        cpu=cpu[order],
        weight=np.asarray(data["photons", "particle_weight"].v, dtype=np.longdouble)[
            order
        ],
        z=np.asarray(data["photons", "particle_position_z"].v, dtype=np.longdouble)[
            order
        ],
        momentum=momentum[order],
        mass=mass[order],
    )


def retained_photons(initial, current):
    initial_pairs = set(zip(initial["ids"].tolist(), initial["cpu"].tolist()))
    current_pairs = set(zip(current["ids"].tolist(), current["cpu"].tolist()))
    retained_pairs = sorted(initial_pairs & current_pairs)
    initial_index = {
        pair: index for index, pair in enumerate(zip(initial["ids"], initial["cpu"]))
    }
    current_index = {
        pair: index for index, pair in enumerate(zip(current["ids"], current["cpu"]))
    }
    index0 = [initial_index[pair] for pair in retained_pairs]
    index = [current_index[pair] for pair in retained_pairs]
    return index0, index, retained_pairs


def check(directory, angular_spread=False):
    paths = plotfiles(directory)
    steps = [int(path.name[5:]) for path in paths]
    if not steps or steps[0] != 0 or steps[-1] != 80:
        raise RuntimeError(f"expected complete step 0..80 history, got {steps}")
    rows = [load(path) for path in paths]
    packets = [photons(path) for path in paths]

    initial = rows[0]
    initial_packets = packets[0]
    live0 = initial_packets["weight"] != 0
    assert np.count_nonzero(live0) > 0
    assert np.all(initial_packets["z"][live0] >= K_ZMIN)
    assert np.all(initial_packets["z"][live0] < K_ZMAX)
    assert initial["field_e"] == 0
    assert initial["energy"] > 0
    photon_momentum0 = initial_packets["momentum"][live0]
    norms0 = np.sqrt(np.sum(photon_momentum0**2, axis=1))
    directions0 = photon_momentum0 / norms0[:, None]
    weights0 = initial_packets["weight"][live0] / np.sum(
        initial_packets["weight"][live0]
    )
    mean_direction0 = np.sum(weights0[:, None] * directions0, axis=0)
    direction_spread0 = np.sqrt(
        np.sum(weights0[:, None] * (directions0 - mean_direction0) ** 2, axis=0)
    )
    if angular_spread:
        assert np.max(np.abs(mean_direction0 - EXPECTED_DIRECTION)) < 0.04
        assert np.min(direction_spread0) > 0.02
    else:
        assert np.max(np.abs(directions0 - EXPECTED_DIRECTION)) < 1e-12
        assert abs(initial["energy"] / 1250 - 1) < 1e-12
    initial_pairs = list(zip(initial_packets["ids"], initial_packets["cpu"]))
    assert len(initial_pairs) == len(set(initial_pairs))

    mass = initial["mass"]
    gamma0 = np.sqrt(1 + np.sum(initial["u"] ** 2, axis=1) / C**2)
    records = []
    later_conversion = False
    later_retained_k = False
    final_recoil = np.zeros(3, dtype=np.longdouble)

    for row, packet in zip(rows, packets):
        assert np.array_equal(row["ids"], initial["ids"])
        assert np.array_equal(row["mass"], mass)
        assert np.all(np.isfinite(row["q"]))
        assert row["field_e"] >= 0

        live = packet["weight"] != 0
        index0, index, retained_pairs = retained_photons(initial_packets, packet)
        if index:
            assert np.array_equal(
                packet["mass"][index], initial_packets["mass"][index0]
            )
            if (
                row["step"] > 0
                and row["field_e"] > 0
                and np.any(
                    (packet["z"][index] < D_ZMIN) | (packet["z"][index] >= D_ZMAX)
                )
            ):
                later_retained_k = True

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
        material_momentum = np.sum(
            mass[:, None] * (du + row["carry"] - initial["carry"]), axis=0
        )
        momentum_error = (row["q"] - initial["q"]) / C + material_momentum
        assert abs(energy_error) < 1e-10 * initial["energy"]
        assert np.max(np.abs(momentum_error)) < 1e-10 * initial["energy"] / C

        if row["step"] > 0 and row["field_e"] > 0:
            later_conversion = True
        if row["step"] == rows[-1]["step"]:
            final_recoil = material_momentum
        records.append(
            dict(
                step=row["step"],
                photons=int(np.count_nonzero(live)),
                retained_initial_photons=len(retained_pairs),
                field_energy_J=float(row["field_e"]),
                photon_energy_J=float(row["photon_e"]),
                ion_kinetic_gain_J=float(kinetic),
                material_recoil_kgms=list(map(float, material_momentum)),
                raw_energy_error_J=float(energy_error),
                raw_momentum_error_kgms=list(map(float, momentum_error)),
            )
        )

    assert later_conversion
    assert later_retained_k
    assert np.all(np.abs(final_recoil) > 0)
    result = dict(
        model="existing whole-vector flux-relaxed runtime; heterogeneous stationary chi probe",
        initial=dict(
            photon_energy_J=float(initial["energy"]),
            mean_direction=list(map(float, mean_direction0)),
            direction_spread=list(map(float, direction_spread0)),
            photons=int(np.count_nonzero(live0)),
            field_energy_J=float(initial["field_e"]),
        ),
        history=records,
    )
    (directory / "flux_relaxed_mixed_probe_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(records[-1], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--angular-spread", action="store_true")
    args = parser.parse_args()
    check(args.directory, args.angular_spread)
