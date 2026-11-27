#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Compare Cartesian multi-species resolved-Qei continuation and restart state."""

import argparse
from pathlib import Path

import numpy as np
import yt

parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path)
args = parser.parse_args()
yt.set_log_level(50)


def load(path):
    dataset = yt.load(str(path))
    grid = dataset.covering_grid(0, dataset.domain_left_edge, dataset.domain_dimensions)
    fields = {
        name: grid["boxlib", name].v
        for name in ("rho", "Te", "hybrid_qei_electron_energy_cumulative_fp")
    }
    data = dataset.all_data()
    particles = {}
    for species in ("light", "heavy"):
        order = np.lexsort(
            (data[species, "particle_cpu"].v, data[species, "particle_id"].v)
        )
        particles[species] = {
            name: data[species, name].v[order]
            for species_name, name in dataset.field_list
            if species_name == species
        }
    return fields, particles


fields, particles = load(Path("diags/plt000002"))
for values in (
    *fields.values(),
    *(array for species in particles.values() for array in species.values()),
):
    assert np.all(np.isfinite(values))
assert np.min(fields["Te"]) > 0
assert np.max(np.abs(fields["hybrid_qei_electron_energy_cumulative_fp"])) > 0
checkpoint = Path("diags/chk000002")
support = (checkpoint / "HybridQeiSupport.txt").read_text().split()
assert support[0] == "resolved_pairwise_v3"
assert int(support[1]) == 1 and int(support[2]) == 2

if args.reference:
    reference_fields, reference_particles = load(args.reference / "diags/plt000002")

    def compare(actual, reference, prefix=""):
        assert actual.keys() == reference.keys()
        for name, values in actual.items():
            expected = reference[name]
            label = f"{prefix}{name}"
            if isinstance(values, dict):
                compare(values, expected, label + "/")
                continue
            scale = max(np.max(np.abs(expected)), np.finfo(float).tiny)
            np.testing.assert_allclose(
                values,
                expected,
                rtol=4.0e-12,
                atol=4.0e-12 * scale,
                err_msg=label,
            )

    compare(fields, reference_fields)
    compare(particles, reference_particles)
    print("Cartesian multi-species resolved-Qei restart is deterministic.")
else:
    initial, _ = load(Path("diags/plt000000"))
    assert np.max(np.abs(fields["Te"] - initial["Te"])) > 0
    assert fields["rho"].shape == initial["rho"].shape
    print("Cartesian multi-species resolved-Qei run exchanged thermal energy.")
