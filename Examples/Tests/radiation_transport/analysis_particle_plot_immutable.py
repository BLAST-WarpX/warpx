#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Check SI output and selection alongside the C++ live-state equality gate."""

import argparse
import json
from pathlib import Path

import numpy as np
import yt


def read(path):
    ds = yt.load(str(path))
    data = ds.all_data()
    values = {
        name: data[species, name].v
        for species, name in ds.field_list
        if species == "ions"
    }
    order = np.lexsort((values["particle_cpu"], values["particle_id"]))
    return {name: value[order] for name, value in values.items()}


parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--openpmd", type=Path)
args = parser.parse_args()
root = args.root
baseline = read(root / "readonly_0_000000")
assert len(baseline["particle_id"]) == 64
# A broad magnitude check distinguishes SI momentum from internal proper velocity.
assert np.all(
    (abs(baseline["particle_momentum_z"]) > 1e-22)
    & (abs(baseline["particle_momentum_z"]) < 2e-22)
)
for suffix in ("ux", "uy", "uz", "work"):
    assert np.all(baseline[f"particle_radiation_impulse_test_{suffix}"] != 0)
for mode, selected in (
    (1, baseline["particle_id"] % 2 == 0),
    (2, baseline["particle_momentum_z"] > 0),
):
    output = read(root / f"readonly_{mode}_000000")
    assert np.count_nonzero(selected) == 32
    assert output.keys() == baseline.keys()
    for name in baseline:
        np.testing.assert_array_equal(
            output[name], baseline[name][selected], err_msg=name
        )
print("SI output, nonzero carry, and uniform/parser particle selections pass")

if args.openpmd:
    for mode in range(3):
        plot = read(root / f"readonly_{mode}_000000")
        path = args.openpmd / f"readonly_{mode}_" / "openpmd_000000.json"
        species = json.loads(path.read_text())["data"]["0"]["particles"]["ions"]
        fields = {"particle_position_x": species["position"]["z"]}
        fields.update(
            {f"particle_momentum_{axis}": species["momentum"][axis] for axis in "xyz"}
        )
        fields.update(
            {
                f"particle_radiation_impulse_test_{suffix}": species[record]
                for suffix, record in (
                    ("ux", "radiationImpulseTestUx"),
                    ("uy", "radiationImpulseTestUy"),
                    ("uz", "radiationImpulseTestUz"),
                    ("work", "radiationImpulseTestWork"),
                )
            }
        )
        fields["particle_weight"] = species["weighting"]
        order = np.argsort(fields["particle_position_x"]["data"])
        plot_order = np.argsort(plot["particle_position_x"])
        for name, component in fields.items():
            assert component["attributes"]["unitSI"]["value"] == 1
            np.testing.assert_array_equal(
                np.asarray(component["data"])[order],
                plot[name][plot_order],
                err_msg=name,
            )
    print("openPMD JSON output matches plotfile momentum, carry and selections exactly")
