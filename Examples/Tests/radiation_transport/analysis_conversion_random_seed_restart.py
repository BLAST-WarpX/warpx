#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check resolved-random-seed persistence across particle conversion restart."""

import argparse
from pathlib import Path

import numpy as np
import yt
from analysis_precision import add_precision_arguments, precision_dtypes
from scipy.constants import c


def particle_records(plotfile: Path) -> np.ndarray:
    dataset = yt.load(str(plotfile))
    if ("photons", "particle_weight") not in dataset.field_list:
        return np.empty((0, 7))
    data = dataset.all_data()
    columns = [
        data["photons", f"particle_position_{axis}"].to_value("m") for axis in "xyz"
    ]
    columns.extend(
        data["photons", f"particle_momentum_{axis}"].to_value("kg*m/s")
        for axis in "xyz"
    )
    columns.append(data["photons", "particle_weight"].to_ndarray())
    records = np.column_stack(columns)
    return records[np.lexsort(records.T[::-1])]


def checkpoint_conversion_seed(checkpoint: Path) -> int:
    tokens = (checkpoint / "RadiationTransport_data.txt").read_text().split()
    assert len(tokens) == 3
    seed = int(tokens[2])
    assert seed > 0
    return seed


parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path)
parser.add_argument("--seed-checkpoint", type=Path)
add_precision_arguments(parser)
args = parser.parse_args()
_, particle_dtype, cross_dtype = precision_dtypes(args)

before_conversion = particle_records(Path("diags/diag1000001"))
after_conversion = particle_records(Path("diags/diag1000002"))
assert before_conversion.shape == (0, 7)
assert after_conversion.shape == (64, 7)
assert np.all(np.isfinite(after_conversion))

photon_energy = 1.0e-15
expected_energy = 8.0e-13
particle_rtol = 3.0e-6 if particle_dtype == np.float32 else 2.0e-12
ledger_rtol = 4.0e-6 if cross_dtype == np.float32 else 3.0e-13
represented_energy = np.sum(
    after_conversion[:, 6] * c * np.linalg.norm(after_conversion[:, 3:6], axis=1)
)
np.testing.assert_allclose(
    np.linalg.norm(after_conversion[:, 3:6], axis=1),
    photon_energy / c,
    rtol=particle_rtol,
)
np.testing.assert_allclose(
    represented_energy,
    expected_energy,
    rtol=ledger_rtol,
)

if args.seed_checkpoint is not None:
    restart_seed = checkpoint_conversion_seed(Path("diags/chk000002"))
    checkpoint_seed = checkpoint_conversion_seed(args.seed_checkpoint)
    assert restart_seed == checkpoint_seed
elif args.reference is None:
    checkpoint_conversion_seed(Path("diags/chk000001"))
else:
    reference_before = particle_records(args.reference / "diags/diag1000001")
    reference_after = particle_records(args.reference / "diags/diag1000002")
    np.testing.assert_array_equal(before_conversion, reference_before)
    np.testing.assert_array_equal(after_conversion, reference_after)
    restart_seed = checkpoint_conversion_seed(Path("diags/chk000002"))
    reference_seed = checkpoint_conversion_seed(args.reference / "diags/chk000001")
    assert restart_seed == reference_seed

print(
    "random-seed conversion restart: "
    f"pre-conversion={len(before_conversion)}, "
    f"post-conversion={len(after_conversion)}, "
    f"energy={represented_energy:.16e} J"
)
