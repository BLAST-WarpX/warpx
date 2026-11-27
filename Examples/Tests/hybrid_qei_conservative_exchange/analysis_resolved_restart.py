#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Compare moving Qei continuation, including particle state and thermal history."""

import argparse
import sys
from pathlib import Path

import numpy as np
import yt

parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path)
parser.add_argument("--implicit", action="store_true")
parser.add_argument("--initial-pressure", type=float)
args = parser.parse_args()
yt.set_log_level(50)


def load(path):
    dataset = yt.load(str(path))
    grid = dataset.covering_grid(0, dataset.domain_left_edge, dataset.domain_dimensions)
    fields = {
        name: grid["boxlib", name].v
        for name in ("rho", "Te", "hybrid_qei_electron_energy_cumulative_fp")
    }
    if args.initial_pressure is not None:
        fields["Pe"] = grid["boxlib", "Pe"].v
    data = dataset.all_data()
    order = np.lexsort((data["ions", "particle_cpu"].v, data["ions", "particle_id"].v))
    particles = {
        name: data["ions", name].v[order]
        for species, name in dataset.field_list
        if species == "ions"
    }
    return fields, particles


fields, particles = load(Path("diags/plt000200"))
for values in (*fields.values(), *particles.values()):
    assert np.all(np.isfinite(values))
assert np.min(fields["Te"]) > 0
assert np.max(np.abs(fields["hybrid_qei_electron_energy_cumulative_fp"])) > 0
checkpoint = Path("diags/chk000200")
support = (checkpoint / "HybridQeiSupport.txt").read_text().split()
assert support[0] == "resolved_pairwise_v3"
assert int(support[1]) == 1 and int(support[2]) == 200
assert (checkpoint / "HybridIdealElectronTransport.txt").read_text().strip() == (
    "ideal_finite_volume_implicit_v1" if args.implicit else "ideal_finite_volume_v1"
)
if args.reference:
    reference_fields, reference_particles = load(args.reference / "diags/plt000200")
    # Same-rank continuation is deterministic. A field-relative absolute floor
    # permits cancellation at zeros without hiding an incorrect nonzero history.
    for actual, reference in (
        (fields, reference_fields),
        (particles, reference_particles),
    ):
        assert actual.keys() == reference.keys()
        for name, values in actual.items():
            expected = reference[name]
            scale = max(np.max(np.abs(expected)), np.finfo(float).tiny)
            np.testing.assert_allclose(
                values, expected, rtol=2.0e-12, atol=2.0e-12 * scale, err_msg=name
            )
    print(
        "Moving resolved-Qei restart preserves fields, particles and cumulative history."
    )
else:
    initial, _ = load(Path("diags/plt000000"))
    if args.initial_pressure is not None:
        sys.path.insert(
            0, str(Path(__file__).resolve().parents[1] / "radiation_transport")
        )
        from analysis_moving_moment_interface import nodes

        charge = nodes(
            Path("diags/plt000000/raw_fields/Level_0/rho_fp_H"),
            [8, 16],
            periodic=[False, True],
        )
        temperature = nodes(
            Path("diags/chk000000/Level_0/hybrid_electron_temperature_fp[level=0]_H"),
            [8, 16],
            periodic=[False, True],
        )
        np.testing.assert_allclose(
            charge * temperature * (1.380649e-23 / 1.602176634e-19),
            args.initial_pressure,
            rtol=256 * np.finfo(float).eps,
            atol=0,
        )
        assert np.ptp(temperature) / np.mean(temperature) > 0.1
    assert np.max(np.abs(fields["rho"] - initial["rho"])) > 0
    assert np.max(np.abs(fields["Te"] - initial["Te"])) > 0
    print("Nonuniform material moves and exchanges nonzero thermal energy.")
