#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Native vacuum crossings: ballistic motion, charge measure and positivity.

The separate M-matrix oracle checks transport energy conservation. This full
native probe still includes the unqualified RZ pressure-work closure, so it
does not pretend that its electron caloric inventory is conserved.
"""

import sys
from pathlib import Path

import numpy as np
import yt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "radiation_transport"))
from analysis_moving_moment_interface import nodes

yt.set_log_level(50)
length, nr, nz = 1.0e-3, 8, 16
volume = 2 * np.pi * np.arange(nr + 1) * (length / nr) ** 2 * (length / nz)
volume[0] = np.pi * (length / nr) ** 2 * (length / nz) / 3
volume[-1] *= 0.5
states = []
for step in (0, 125):
    plot = Path(f"diags/plt{step:06d}")
    dataset = yt.load(str(plot))
    data = dataset.all_data()
    order = np.lexsort((data["ions", "particle_cpu"].v, data["ions", "particle_id"].v))
    particle = {
        name: data["ions", name].v[order]
        for species, name in dataset.field_list
        if species == "ions"
    }
    rho = nodes(plot / "raw_fields/Level_0/rho_fp_H", [nr, nz], periodic=[False, True])
    temperature = nodes(
        Path(f"diags/chk{step:06d}/Level_0/")
        / "hybrid_electron_temperature_fp[level=0]_H",
        [nr, nz],
        periodic=[False, True],
    )
    assert np.count_nonzero(rho == 0) > rho.size / 4
    assert np.min(rho) >= 0 and np.min(temperature[rho > 0]) > 0
    charge = np.sum(rho * volume[:, None])
    particle_charge = 1.602176634e-19 * np.sum(particle["particle_weight"])
    np.testing.assert_allclose(charge, particle_charge, rtol=1.0e-12, atol=0)
    states.append(
        (dataset, particle, charge, np.sum(rho * temperature * volume[:, None]))
    )
first, last = states
for name in (
    "particle_weight",
    "particle_id",
    "particle_cpu",
    "particle_position_x",
    "particle_momentum_x",
    "particle_momentum_y",
    "particle_momentum_z",
):
    np.testing.assert_allclose(
        last[1][name],
        first[1][name],
        rtol=128 * np.finfo(float).eps,
        atol=0,
        err_msg=name,
    )
speed = 299792458.0 * 1.0e-3 / np.sqrt(1 + 1.0e-6)
expected = (
    first[1]["particle_position_y"] + speed * float(last[0].current_time)
) % length
np.testing.assert_allclose(
    last[1]["particle_position_y"],
    expected,
    rtol=0,
    atol=256 * np.finfo(float).eps * length,
)
np.testing.assert_allclose(last[2], first[2], rtol=1.0e-12, atol=0)
print(
    "125 steps of periodic vacuum crossings preserve ballistic particles, charge and positivity."
)
print(
    "Separate, unqualified pressure-work caloric change:", float(last[3] / first[3] - 1)
)
