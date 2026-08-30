#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check persistence of a below-ULP material-realization residual."""

import argparse
import re
from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer

FIELD_NAMES = (
    "rho",
    "Te",
    "Pe",
    "radiation_material_energy",
    "radiation_diffusion_energy",
)


def load_plotfile(path: Path) -> dict[str, np.ndarray]:
    with (path / "Header").open() as header:
        header.readline()
        num_fields = int(header.readline())
        names = [header.readline().strip() for _ in range(num_fields)]
    values = _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), names)
    return {
        name: np.asarray(value, dtype=np.float64).squeeze()
        for name, value in values.items()
    }


def load_table(path: Path) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path))


def row_at_step(table: np.ndarray, step: int) -> np.ndarray:
    rows = table[np.asarray(table[:, 0], dtype=int) == step]
    assert rows.shape[0] >= 1, (step, rows)
    return rows[-1]


def plotfile_at_step(step: int) -> Path:
    candidates = []
    for path in Path("diags").glob("diag*"):
        if not (path.is_dir() and (path / "Header").is_file()):
            continue
        match = re.fullmatch(r"diag(\d+)", path.name)
        if match is None:
            continue
        suffix = match.group(1)
        candidates.append((int(suffix[-6:]), path))
    matches = [path for candidate_step, path in candidates if candidate_step == step]
    assert len(matches) == 1, (step, candidates)
    return matches[0]


def assert_fields_finite(fields: dict[str, np.ndarray]) -> None:
    for name in FIELD_NAMES:
        assert fields[name].shape == (8,)
        assert np.all(np.isfinite(fields[name]))


def checkpoint_values(path: Path) -> tuple[np.ndarray, np.ndarray]:
    transport_path = path / "RadiationTransport_data.txt"
    reduced_path = path / "radiation_energy_RadiationEnergy_data.txt"
    assert transport_path.is_file()
    assert reduced_path.is_file()
    transport = np.asarray(
        [float(token) for token in transport_path.read_text().split()]
    )
    reduced = np.asarray([float(token) for token in reduced_path.read_text().split()])
    assert transport.shape == (2,)
    assert reduced.shape == (6,)
    assert np.all(np.isfinite(transport))
    assert np.all(np.isfinite(reduced))
    return transport, reduced


parser = argparse.ArgumentParser()
parser.add_argument("--compare-reference", action="store_true")
parser.add_argument("--precision", choices=["SINGLE", "DOUBLE"], required=True)
args = parser.parse_args()

precision = np.float32 if args.precision == "SINGLE" else np.float64
precision_eps = np.finfo(precision).eps

radiation_table = load_table(Path("diags/radiation_energy.txt"))
assert radiation_table.shape[1] >= 17
step_two = row_at_step(radiation_table, 2) if not args.compare_reference else None
step_three = row_at_step(radiation_table, 3)
final_fields = load_plotfile(plotfile_at_step(3))
assert_fields_finite(final_fields)

if not args.compare_reference:
    initial_fields = load_plotfile(plotfile_at_step(1))
    step_two_fields = load_plotfile(plotfile_at_step(2))
    assert_fields_finite(initial_fields)
    assert_fields_finite(step_two_fields)
    step_one = row_at_step(radiation_table, 1)

    table_scale = max(np.max(np.abs(radiation_table[:, 2:])), np.finfo(np.float64).tiny)
    table_atol = 256.0 * precision_eps * table_scale
    q_two = step_one[2] - step_two[2]
    d_two = step_two[5]
    r_two = step_two[15]
    residual_scale = max(abs(q_two), abs(d_two), abs(r_two), np.finfo(np.float64).tiny)
    residual_atol = 256.0 * precision_eps * residual_scale
    assert q_two < 0.0
    assert abs(r_two) > max(0.5 * abs(q_two), residual_atol)
    np.testing.assert_allclose(r_two, q_two - d_two, rtol=0.0, atol=residual_atol)
    np.testing.assert_allclose(step_two[16], r_two, rtol=0.0, atol=residual_atol)

    q_three = step_two[2] - step_three[2]
    d_three = step_three[5]
    np.testing.assert_allclose(q_three, 0.0, rtol=0.0, atol=residual_atol)
    np.testing.assert_allclose(d_three, 0.0, rtol=0.0, atol=residual_atol)
    np.testing.assert_allclose(
        step_three[15], q_three - d_three, rtol=0.0, atol=residual_atol
    )
    np.testing.assert_allclose(
        step_three[16], step_two[16], rtol=0.0, atol=residual_atol
    )
    np.testing.assert_allclose(step_three[15], 0.0, rtol=0.0, atol=residual_atol)
    np.testing.assert_allclose(step_three[5], 0.0, rtol=0.0, atol=residual_atol)
    np.testing.assert_allclose(step_three[6], step_two[6], rtol=0.0, atol=table_atol)
    np.testing.assert_allclose(step_three[9], 0.0, rtol=0.0, atol=table_atol)
    np.testing.assert_allclose(step_three[10], 0.0, rtol=0.0, atol=table_atol)
    np.testing.assert_allclose(step_three[2], step_two[2], rtol=0.0, atol=table_atol)
    np.testing.assert_allclose(step_two[7:9], 0.0, rtol=0.0, atol=table_atol)
    np.testing.assert_allclose(step_three[7:9], 0.0, rtol=0.0, atol=table_atol)

    transport, reduced = checkpoint_values(Path("diags/chk000002"))
    np.testing.assert_allclose(transport[1], step_two[16], rtol=0.0, atol=residual_atol)
    np.testing.assert_allclose(reduced[0], step_two[6], rtol=0.0, atol=table_atol)
    np.testing.assert_allclose(reduced[5], step_two[16], rtol=0.0, atol=residual_atol)
    assert int(reduced[4]) == 1
else:
    reference_dir = Path.cwd().with_name(Path.cwd().name.removesuffix("_restart"))
    reference_table = load_table(reference_dir / "diags/radiation_energy.txt")
    reference_step_two = row_at_step(reference_table, 2)
    reference_step_three = row_at_step(reference_table, 3)
    reference_plotfile = reference_dir / "diags" / plotfile_at_step(3).name
    reference_fields = load_plotfile(reference_plotfile)
    # The restart table contains the continued step.  Its cumulative residual
    # must match the producer's checkpointed step-2 value and uninterrupted
    # step-3 value, even though the current step-3 residual is zero.
    table_scale = max(
        np.max(np.abs(radiation_table[:, 2:])),
        np.max(np.abs(reference_table[:, 2:])),
        np.finfo(np.float64).tiny,
    )
    table_atol = 256.0 * precision_eps * table_scale
    residual_scale = max(
        abs(step_three[15]),
        abs(step_three[16]),
        abs(reference_step_two[15]),
        abs(reference_step_two[16]),
        abs(reference_step_three[15]),
        abs(reference_step_three[16]),
        np.finfo(np.float64).tiny,
    )
    residual_atol = 256.0 * precision_eps * residual_scale
    np.testing.assert_allclose(
        step_three, reference_step_three, rtol=0.0, atol=table_atol
    )
    np.testing.assert_allclose(step_three[15], 0.0, rtol=0.0, atol=residual_atol)
    np.testing.assert_allclose(
        step_three[16], reference_step_two[16], rtol=0.0, atol=residual_atol
    )
    np.testing.assert_allclose(
        step_three[16], reference_step_three[16], rtol=0.0, atol=residual_atol
    )

    field_scale = max(
        *(np.max(np.abs(final_fields[name])) for name in FIELD_NAMES),
        *(np.max(np.abs(reference_fields[name])) for name in FIELD_NAMES),
        np.finfo(np.float64).tiny,
    )
    field_atol = 256.0 * precision_eps * field_scale
    for name in FIELD_NAMES:
        np.testing.assert_allclose(
            final_fields[name], reference_fields[name], rtol=0.0, atol=field_atol
        )

print(
    "1D hybrid realization restart: "
    f"step3_R={step_three[15]:.16e} J, "
    f"step3_cumulative_R={step_three[16]:.16e} J"
)
