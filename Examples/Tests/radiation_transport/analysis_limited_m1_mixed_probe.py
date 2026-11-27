#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Reuse the mixed-front conservation gate and audit same-rank native restart."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import yt
from analysis_flux_relaxed_mixed_probe import check, plotfiles
from analysis_flux_relaxed_roundtrip import MP, C, load

yt.funcs.mylog.setLevel(40)


def compare_values(left, right, label, arithmetic_scale=0):
    left = np.asarray(left, dtype=np.longdouble)
    right = np.asarray(right, dtype=np.longdouble)
    assert left.shape == right.shape, label
    assert np.all(np.isfinite(left)) and np.all(np.isfinite(right)), label
    # Existing roundtrip restart relative bound; carry/coordinates additionally
    # use the existing 128-epsilon arithmetic bound on uncancelled operands.
    bound = 1e-11 * np.maximum(abs(left), abs(right))
    bound += 128 * np.finfo(np.float64).eps * arithmetic_scale
    difference = abs(left - right)
    assert np.all(difference <= bound), (label, np.max(difference), np.max(bound))
    return float(np.max(difference, initial=0))


def compare_final(reference, restarted, checkpoint_plot):
    before = yt.load(str(reference))
    after = yt.load(str(restarted))
    assert before.current_time == after.current_time
    assert before.field_list == after.field_list
    records = {}
    checkpoint_data = yt.load(str(checkpoint_plot)).all_data()
    checkpoint_ids = set(
        zip(
            checkpoint_data["photons", "particle_id"].v,
            checkpoint_data["photons", "particle_cpu"].v,
        )
    )
    for species in ("ions", "photons"):
        old_data, new_data = before.all_data(), after.all_data()
        orders = []
        for data in (old_data, new_data):
            orders.append(
                np.lexsort(
                    (data[species, "particle_cpu"].v, data[species, "particle_id"].v)
                )
            )
        if species == "photons":
            survivors = []
            for data in (old_data, new_data):
                identities = list(
                    zip(data[species, "particle_id"].v, data[species, "particle_cpu"].v)
                )
                assert len(identities) == len(set(identities)), "duplicate photon IDs"
                survivors.append(set(identities) & checkpoint_ids)
            assert survivors[0] == survivors[1], (
                "checkpoint-surviving photon identities changed"
            )
            records["checkpoint_surviving_photon_count"] = len(survivors[0])
            # Birth allocation IDs are not bitwise restart identities, even at
            # the same MPI size. Match all physical photon records canonically.
            physical_names = ["particle_position_" + axis for axis in "xyz"]
            physical_names += ["particle_momentum_" + axis for axis in "xyz"]
            physical_names += ["particle_weight"]
            survivor_orders = []
            for data in (old_data, new_data):
                indices = {
                    identity: index
                    for index, identity in enumerate(
                        zip(
                            data[species, "particle_id"].v,
                            data[species, "particle_cpu"].v,
                        )
                    )
                }
                survivor_orders.append(
                    [indices[identity] for identity in sorted(survivors[0])]
                )
            for name in physical_names:
                left = old_data[species, name].v[survivor_orders[0]]
                right = new_data[species, name].v[survivor_orders[1]]
                scale = (
                    before.domain_width["xyz".index(name[-1])].v
                    if "position_" in name
                    else 0
                )
                records["checkpoint_survivors/" + name] = compare_values(
                    left, right, name, scale
                )
            orders = [
                np.lexsort(
                    np.stack([data[species, name].v for name in physical_names])[::-1]
                )
                for data in (old_data, new_data)
            ]
        names = sorted(name for kind, name in before.field_list if kind == species)
        velocity_scale = np.zeros(3, dtype=np.longdouble)
        kinetic_scale = np.longdouble(0)
        if species == "ions":
            for data in (old_data, new_data):
                u = np.stack(
                    [
                        np.asarray(
                            data[species, "particle_momentum_" + axis].v,
                            dtype=np.longdouble,
                        )
                        / MP
                        for axis in "xyz"
                    ],
                    axis=1,
                )
                velocity_scale = np.maximum(velocity_scale, np.max(abs(u), axis=0))
                u2 = np.sum(u * u, axis=1)
                kinetic_scale = max(
                    kinetic_scale, np.max(u2 / (np.sqrt(1 + u2 / C**2) + 1))
                )
        for name in names:
            left = old_data[species, name].v[orders[0]]
            right = new_data[species, name].v[orders[1]]
            label = species + "/" + name
            if name in ("particle_id", "particle_cpu"):
                if species == "photons":
                    records[label + "_bitwise_restart"] = bool(
                        np.array_equal(left, right)
                    )
                    continue
                np.testing.assert_array_equal(left, right, err_msg=label)
                records[label] = 0.0
                continue
            arithmetic_scale = np.longdouble(0)
            if "position_" in name:
                arithmetic_scale = before.domain_width["xyz".index(name[-1])].v
            elif "radiation_impulse_" in name:
                arithmetic_scale = (
                    kinetic_scale
                    if name.endswith("work")
                    else (velocity_scale["xyz".index(name[-1])])
                )
            records[label] = compare_values(left, right, label, arithmetic_scale)
    old_grid = before.covering_grid(
        0, before.domain_left_edge, before.domain_dimensions
    )
    new_grid = after.covering_grid(0, after.domain_left_edge, after.domain_dimensions)
    for field in before.field_list:
        if field[0] == "boxlib":
            records["/".join(field)] = compare_values(
                old_grid[field].v, new_grid[field].v, "/".join(field)
            )
    return records


def run(directory, reference=None):
    directory = directory.resolve()
    if reference is None:
        check(directory)
        report = json.loads(
            (directory / "flux_relaxed_mixed_probe_results.json").read_text()
        )
    else:
        reference = reference.resolve()
        baseline_paths, restart_paths = plotfiles(reference), plotfiles(directory)
        assert baseline_paths and restart_paths
        restart_start = int(restart_paths[0].name[5:])
        assert 40 <= restart_start <= 80
        assert int(restart_paths[-1].name[5:]) == 80
        # Feed the unchanged full-history gate a temporary view: pre-restart
        # baseline snapshots plus the actual restart continuation snapshots.
        with tempfile.TemporaryDirectory(
            prefix="limited_m1_analysis_", dir=directory
        ) as temporary:
            merged = Path(temporary)
            (merged / "diags").mkdir()
            for path in baseline_paths:
                if int(path.name[5:]) < restart_start:
                    (merged / "diags" / path.name).symlink_to(
                        path.resolve(), target_is_directory=True
                    )
            for path in restart_paths:
                (merged / "diags" / path.name).symlink_to(
                    path.resolve(), target_is_directory=True
                )
            check(merged)
            report = json.loads(
                (merged / "flux_relaxed_mixed_probe_results.json").read_text()
            )
        baseline, final = load(baseline_paths[-1]), load(restart_paths[-1])
        initial_energy = load(baseline_paths[0])["energy"]
        assert abs(final["energy"] - baseline["energy"]) < 1e-11 * initial_energy
        assert np.max(abs(final["q"] - baseline["q"])) < 1e-11 * initial_energy
        assert np.max(abs(final["u"] - baseline["u"])) < 1e-8
        report["restart_attribute_maximum_differences"] = compare_final(
            baseline_paths[-1],
            restart_paths[-1],
            next(path for path in baseline_paths if int(path.name[5:]) == 40),
        )
        report["photon_identity_contract"] = (
            "Unique final IDs and exact checkpoint-survivor identity sets; newly born "
            "photons compared by all physical records, not bitwise allocation IDs."
        )
    report["model"] = "explicit limited M1 transport plus finite-mass coherent source"
    (directory / "limited_m1_mixed_probe_results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference", type=Path)
    arguments = parser.parse_args()
    run(arguments.directory, arguments.reference)
