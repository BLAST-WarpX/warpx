#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse
from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer

parser = argparse.ArgumentParser()
parser.add_argument("--coupling", choices=["hybrid", "kinetic"], required=True)
parser.add_argument("--multigroup", action="store_true")
args = parser.parse_args()


def load_reduced_table(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open() as stream:
        labels = [
            token.split("]", 1)[1]
            for token in stream.readline().lstrip("#").split()
            if token.startswith("[") and "]" in token
        ]
    data = np.atleast_2d(np.loadtxt(path))
    assert len(labels) == data.shape[1]
    return labels, data


def load_particle_dtype(plotfile: Path) -> np.dtype:
    particle_precisions = set()
    for header_path in sorted(plotfile.glob("*/Header")):
        with header_path.open() as header:
            version = header.readline().strip().lower()
        if version.endswith("_single"):
            particle_precisions.add(np.dtype(np.float32))
        elif version.endswith("_double"):
            particle_precisions.add(np.dtype(np.float64))
    assert len(particle_precisions) == 1
    return particle_precisions.pop()


def sum_in_dtype(fields: list[np.ndarray], dtype: np.dtype) -> np.generic:
    total = dtype.type(0.0)
    for field in fields:
        total = dtype.type(total + np.sum(field, dtype=dtype))
    return total


plotfile = Path("diags/diag1000020")
with open(plotfile / "Header") as header:
    header.readline()
    n_fields = int(header.readline())
    field_names = [header.readline().strip() for _ in range(n_fields)]

fields = _read_buffer(str(plotfile), str(plotfile / "Level_0" / "Cell_H"), field_names)
material_energy = fields["radiation_material_energy"]
field_dtype = np.dtype(material_energy.dtype)
field_epsilon = np.finfo(field_dtype).eps
particle_dtype = load_particle_dtype(plotfile)
particle_epsilon = np.finfo(particle_dtype).eps
ledger_epsilon = max(field_epsilon, particle_epsilon)
ledger_coefficient = 1000.0 * ledger_epsilon
if args.multigroup:
    diffusion_fields = [
        fields[f"radiation_diffusion_energy_g{group}"] for group in range(4)
    ]
else:
    diffusion_fields = [fields["radiation_diffusion_energy"]]
diffusion_energy = sum_in_dtype(diffusion_fields, field_dtype)
particle_energy = np.loadtxt("diags/particle_energy.txt")
radiation_labels, radiation_energy = load_reduced_table(
    Path("diags/radiation_energy.txt")
)
radiation_active = np.asarray(radiation_energy, dtype=field_dtype)
particle_active = np.asarray(particle_energy, dtype=field_dtype)
radiation_columns = {name: radiation_labels.index(name) for name in radiation_labels}
total = radiation_active[:, radiation_columns["total_radiation(J)"]]
streaming = radiation_active[:, radiation_columns["streaming_photons(J)"]]
diffusion = radiation_active[:, radiation_columns["diffusion_radiation(J)"]]
material = radiation_active[:, radiation_columns["material_exchange(J)"]]
cumulative_material = radiation_active[
    :, radiation_columns["cumulative_material_exchange(J)"]
]
boundary = radiation_active[:, radiation_columns["boundary_energy_loss(J)"]]
cumulative_boundary = radiation_active[
    :, radiation_columns["cumulative_boundary_energy_loss(J)"]
]
residual = radiation_active[:, radiation_columns["numerical_energy_residual(J)"]]
cumulative_residual = radiation_active[
    :, radiation_columns["cumulative_numerical_energy_residual(J)"]
]

with open("diags/particle_energy.txt") as reduced_diag:
    labels = reduced_diag.readline().split()
photon_column = next(
    index for index, label in enumerate(labels) if label.endswith("photons(J)")
)

print(f"final diffusion energy: {diffusion_energy:.16e} J")
print(f"final photon energy:    {particle_active[-1, photon_column]:.16e} J")

assert np.all(np.isfinite(material_energy))
assert all(np.all(np.isfinite(field)) for field in diffusion_fields)
assert all(np.all(field >= 0.0) for field in diffusion_fields)
assert diffusion_energy > 0.0
assert particle_active[-1, photon_column] > particle_active[0, photon_column]
np.testing.assert_allclose(
    total[-1],
    streaming[-1] + diffusion[-1],
    rtol=ledger_coefficient,
)
np.testing.assert_allclose(
    streaming[-1], particle_active[-1, photon_column], rtol=ledger_coefficient
)
np.testing.assert_allclose(diffusion[-1], diffusion_energy, rtol=ledger_coefficient)
np.testing.assert_allclose(
    material[-1], np.sum(material_energy, dtype=field_dtype), rtol=ledger_coefficient
)
np.testing.assert_allclose(
    cumulative_material[-1] - cumulative_material[0],
    np.sum(material[1:], dtype=field_dtype),
    rtol=ledger_coefficient,
)
np.testing.assert_allclose(
    cumulative_boundary[-1] - cumulative_boundary[0],
    np.sum(boundary[1:], dtype=field_dtype),
    rtol=ledger_coefficient,
)
if args.multigroup:
    group_columns = [
        index
        for index, label in enumerate(radiation_labels)
        if label.startswith("diffusion_radiation_group_")
    ]
    np.testing.assert_allclose(
        radiation_active[-1, group_columns],
        [np.sum(field, dtype=field_dtype) for field in diffusion_fields],
        rtol=ledger_coefficient,
    )
    np.testing.assert_allclose(
        diffusion[-1],
        np.sum(radiation_active[-1, group_columns], dtype=field_dtype),
        rtol=ledger_coefficient,
    )
# The signed numerical residual is part of the total cumulative closure. Keep
# its current and cumulative forms tied to the same per-step energy balance.
current_scale = np.maximum.reduce(
    [np.abs(total[:-1]), np.abs(total[1:]), np.abs(material[1:])]
)
cumulative_scale = np.maximum(np.abs(total[0]), np.abs(total[1:]))
cumulative_scale = np.maximum(
    cumulative_scale, np.abs(cumulative_material[1:] - cumulative_material[0])
)
cumulative_scale = np.maximum(
    cumulative_scale, np.abs(cumulative_boundary[1:] - cumulative_boundary[0])
)
cumulative_scale = np.maximum(
    cumulative_scale,
    np.cumsum(current_scale, dtype=field_dtype),
)
tiny = np.finfo(field_dtype).tiny
current_atol = np.maximum(tiny, ledger_coefficient * current_scale)
cumulative_atol = np.maximum(tiny, ledger_coefficient * cumulative_scale)
assert np.all(boundary[1:] >= 0.0)
current_balance = total[:-1] - total[1:] - material[1:] - boundary[1:]
assert np.all(np.abs(residual[1:] - current_balance) <= current_atol)
cumulative_residual_increment = cumulative_residual[1:] - cumulative_residual[0]
cumulative_balance = (
    total[0]
    - total[1:]
    - (cumulative_material[1:] - cumulative_material[0])
    - (cumulative_boundary[1:] - cumulative_boundary[0])
)
assert np.all(
    np.abs(cumulative_residual_increment - cumulative_balance) <= cumulative_atol
)
assert np.all(
    np.abs(cumulative_residual_increment - np.cumsum(residual[1:], dtype=field_dtype))
    <= cumulative_atol
)
global_closure = total[-1] - total[0]
global_closure = field_dtype.type(
    global_closure + cumulative_material[-1] - cumulative_material[0]
)
global_closure = field_dtype.type(
    global_closure + cumulative_boundary[-1] - cumulative_boundary[0]
)
global_closure = field_dtype.type(
    global_closure + cumulative_residual[-1] - cumulative_residual[0]
)
np.testing.assert_allclose(
    global_closure,
    0.0,
    rtol=0.0,
    atol=float(cumulative_atol[-1]),
)

if args.coupling == "hybrid":
    assert np.all(np.isfinite(fields["Te"]))
    assert np.all(fields["Te"] >= 0.0)
else:
    electron_column = next(
        index for index, label in enumerate(labels) if label.endswith("electrons(J)")
    )
    assert particle_active[-1, electron_column] >= 0.0
    assert particle_active[-1, electron_column] < particle_active[0, electron_column]
