#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Check output units, runtime attributes, and unchanged filter selections."""

import numpy as np
import yt


def read(path):
    ds = yt.load(path)
    data = ds.all_data()
    values = {
        name: data[species, name].v
        for species, name in ds.field_list
        if species == "ions"
    }
    order = np.lexsort((values["particle_cpu"], values["particle_id"]))
    return {name: value[order] for name, value in values.items()}


baseline = read("readonly_0_000000")
assert len(baseline["particle_id"]) == 64
assert np.all(
    (abs(baseline["particle_momentum_z"]) > 1e-22)
    & (abs(baseline["particle_momentum_z"]) < 2e-22)
)
assert np.all(baseline["particle_sentinel"] != 0)
for mode, selected in (
    (1, baseline["particle_id"] % 2 == 0),
    (2, baseline["particle_momentum_z"] > 0),
):
    output = read(f"readonly_{mode}_000000")
    assert np.count_nonzero(selected) == 32
    assert output.keys() == baseline.keys()
    for name in baseline:
        np.testing.assert_array_equal(
            output[name], baseline[name][selected], err_msg=name
        )
print("SI output and uniform/parser selections pass")
