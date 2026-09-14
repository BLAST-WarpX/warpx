#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Sustained sub-ULP radiation impulse and checkpoint/restart regression."""

import argparse
from pathlib import Path

import numpy as np
import yt
from scipy.constants import c, m_p


def load_table(path):
    with path.open() as stream:
        labels = [
            token.split("]", 1)[1]
            for token in stream.readline().lstrip("#").split()
            if token.startswith("[") and "]" in token
        ]
    data = np.atleast_2d(np.loadtxt(path))
    assert data.shape[1] == len(labels)
    step_column = labels.index("step()")
    repeated_run = np.flatnonzero(np.diff(data[:, step_column]) <= 0)
    if repeated_run.size:
        data = data[repeated_run[-1] + 1 :]
    return {label: data[:, column] for column, label in enumerate(labels)}


def particle_inventories(path):
    data = yt.load(str(path)).all_data()
    weights = np.asarray(data["ions", "particle_weight"].v, dtype=np.float64)
    momenta = np.stack(
        [
            np.asarray(
                data["ions", f"particle_momentum_{axis}"].to_value("kg*m/s"),
                dtype=np.float64,
            )
            for axis in ("x", "y", "z")
        ],
        axis=1,
    )
    momentum = np.sum(weights[:, None] * momenta, axis=0)
    momentum_squared = np.sum(momenta * momenta, axis=1)
    kinetic_per_particle = (
        momentum_squared
        / m_p
        / (1.0 + np.sqrt(1.0 + momentum_squared / (m_p * c) ** 2))
    )
    return momentum, float(np.sum(weights * kinetic_per_particle))


parser = argparse.ArgumentParser()
parser.add_argument("--compare-reference", action="store_true")
args = parser.parse_args()

energy = load_table(Path("diags/radiation_energy.txt"))
momentum = load_table(Path("diags/radiation_momentum.txt"))
steps = energy["step()"].astype(int)
np.testing.assert_array_equal(steps, momentum["step()"].astype(int))
assert steps[-1] == 400

required_pending = {
    f"pending_{source}_material_{axis}(kg*m/s)"
    for source in ("streaming", "diffusion")
    for axis in ("x", "y", "z")
}
assert required_pending <= momentum.keys()
assert all(np.all(np.isfinite(values)) for values in energy.values())
assert all(np.all(np.isfinite(values)) for values in momentum.values())

assert not np.any(
    np.stack(
        [
            momentum[f"pending_streaming_material_{axis}(kg*m/s)"]
            for axis in ("x", "y", "z")
        ],
        axis=1,
    )
)
assert not np.any(
    np.stack(
        [momentum[f"pending_diffusion_material_{axis}(kg*m/s)"] for axis in ("x", "y")],
        axis=1,
    )
)
pending_z = momentum["pending_diffusion_material_z(kg*m/s)"]
step_149 = np.flatnonzero(steps == 149)
assert step_149.size == 1
assert pending_z[step_149[0]] != 0.0

cumulative_applied = momentum["cumulative_material_z(kg*m/s)"]
momentum_scale = max(float(np.max(np.abs(cumulative_applied))), np.finfo(float).tiny)
assert float(np.max(np.abs(pending_z))) <= 1.0e-10 * momentum_scale

radiation = energy["total_radiation(J)"]
cumulative_material = energy["cumulative_material_exchange(J)"]
cumulative_boundary = energy["cumulative_boundary_energy_loss(J)"]
cumulative_residual = energy["cumulative_numerical_energy_residual(J)"]
initial_conserved_energy = float(
    radiation[0]
    + cumulative_material[0]
    + cumulative_boundary[0]
    + cumulative_residual[0]
)
energy_scale = max(abs(initial_conserved_energy), np.finfo(float).tiny)
np.testing.assert_allclose(
    initial_conserved_energy - radiation - cumulative_material - cumulative_boundary,
    cumulative_residual,
    rtol=0.0,
    atol=1.0e-10 * energy_scale,
)

particle_momentum, particle_kinetic = particle_inventories(Path("diags/diag000400"))
reference_dir = Path.cwd().with_name(Path.cwd().name.removesuffix("_restart"))
reference_energy = (
    load_table(reference_dir / "diags/radiation_energy.txt")
    if args.compare_reference
    else energy
)
cumulative_work = float(np.sum(reference_energy["material_kinetic_exchange(J)"]))
np.testing.assert_allclose(
    particle_momentum[2], cumulative_applied[-1], rtol=1.0e-10, atol=1.0e-20
)
np.testing.assert_allclose(particle_momentum[:2], 0.0, rtol=0.0, atol=1.0e-20)
np.testing.assert_allclose(
    particle_kinetic, cumulative_work, rtol=1.0e-10, atol=1.0e-20
)

if args.compare_reference:
    reference_momentum = load_table(reference_dir / "diags/radiation_momentum.txt")
    for table, reference in (
        (energy, reference_energy),
        (momentum, reference_momentum),
    ):
        reference_steps = reference["step()"].astype(int)
        for row, step in enumerate(steps):
            # The diagnostic written immediately on restart deliberately has
            # zero per-step source fields while retaining checkpointed state
            # and cumulative ledgers. Compare evolved rows from the next step.
            if row == 0:
                continue
            reference_row = np.flatnonzero(reference_steps == step)
            assert reference_row.size == 1
            for label, values in table.items():
                np.testing.assert_allclose(
                    values[row],
                    reference[label][reference_row[0]],
                    rtol=2.0e-12,
                    atol=1.0e-20,
                )
else:
    checkpoint = Path("diags/chk000149/Level_0")
    assert checkpoint.is_dir()
    assert list(checkpoint.glob("radiation_streaming_momentum_carry*"))
    assert list(checkpoint.glob("radiation_diffusion_momentum_carry*"))
