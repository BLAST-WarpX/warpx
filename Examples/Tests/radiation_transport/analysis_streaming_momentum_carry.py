#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Validate repeated sub-ULP streaming recoil and delayed signed work."""

import argparse
from pathlib import Path

import numpy as np
import yt

C_LIGHT = 299792458.0
ION_MASS_KG = 1.0

parser = argparse.ArgumentParser()
parser.add_argument("--lte-emission", action="store_true")
args = parser.parse_args()


def load_table(path: Path) -> dict[str, np.ndarray]:
    with path.open() as stream:
        labels = [
            token.split("]", 1)[1]
            for token in stream.readline().lstrip("#").split()
            if token.startswith("[") and "]" in token
        ]
    data = np.atleast_2d(np.loadtxt(path))
    assert data.shape[1] == len(labels)
    return {label: data[:, column] for column, label in enumerate(labels)}


def particle_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = yt.load(str(path)).all_data()
    weights = np.asarray(data["ions", "particle_weight"].v, dtype=np.float64)
    momentum = np.stack(
        [
            np.asarray(
                data["ions", f"particle_momentum_{axis}"].to_value("kg*m/s"),
                dtype=np.float64,
            )
            for axis in "xyz"
        ],
        axis=1,
    )
    assert weights.shape == (8,)
    assert momentum.shape == (8, 3)
    return weights, momentum


energy = load_table(Path("diags/radiation_energy.txt"))
momentum = load_table(Path("diags/radiation_momentum.txt"))
steps = energy["step()"].astype(int)
np.testing.assert_array_equal(steps, np.arange(33))
np.testing.assert_array_equal(steps, momentum["step()"].astype(int))
assert all(np.all(np.isfinite(values)) for values in energy.values())
assert all(np.all(np.isfinite(values)) for values in momentum.values())

radiation = energy["total_radiation(J)"]
streaming = energy["streaming_photons(J)"]
diffusion = energy["diffusion_radiation(J)"]
material = energy["material_exchange(J)"]
cumulative_material = energy["cumulative_material_exchange(J)"]
internal = energy["material_internal_exchange(J)"]
kinetic = energy["material_kinetic_exchange(J)"]
boundary = energy["boundary_energy_loss(J)"]
cumulative_boundary = energy["cumulative_boundary_energy_loss(J)"]
residual = energy["numerical_energy_residual(J)"]
cumulative_residual = energy["cumulative_numerical_energy_residual(J)"]

energy_scale = float(radiation[0])
energy_atol = 1024.0 * np.finfo(np.float64).eps * energy_scale
np.testing.assert_allclose(streaming + diffusion, radiation, rtol=0.0, atol=energy_atol)
if args.lte_emission:
    assert diffusion[0] == 0.0
    assert np.all(np.diff(diffusion) > 0.0)
    assert diffusion[-1] > 1.0e6 * energy_atol
else:
    np.testing.assert_allclose(diffusion, 0.0, rtol=0.0, atol=energy_atol)
np.testing.assert_allclose(boundary, 0.0, rtol=0.0, atol=energy_atol)
np.testing.assert_allclose(cumulative_boundary, 0.0, rtol=0.0, atol=energy_atol)
np.testing.assert_allclose(
    energy["streaming_boundary_energy_loss(J)"],
    0.0,
    rtol=0.0,
    atol=energy_atol,
)
np.testing.assert_allclose(
    energy["cumulative_streaming_boundary_energy_loss(J)"],
    0.0,
    rtol=0.0,
    atol=energy_atol,
)
np.testing.assert_allclose(
    energy["diffusion_boundary_energy_loss(J)"],
    0.0,
    rtol=0.0,
    atol=energy_atol,
)
np.testing.assert_allclose(
    energy["cumulative_diffusion_boundary_energy_loss(J)"],
    0.0,
    rtol=0.0,
    atol=energy_atol,
)

current_absorption = streaming[:-1] - streaming[1:]
assert np.all(current_absorption > 0.0)
np.testing.assert_allclose(
    material,
    internal + kinetic,
    rtol=2.0e-11,
    atol=energy_atol,
)
np.testing.assert_allclose(
    cumulative_material,
    np.cumsum(material),
    rtol=2.0e-11,
    atol=energy_atol,
)
np.testing.assert_allclose(
    cumulative_residual,
    np.cumsum(residual),
    rtol=0.0,
    atol=energy_atol,
)
calculated_residual = np.zeros_like(residual)
calculated_residual[1:] = radiation[:-1] - radiation[1:] - material[1:] - boundary[1:]
np.testing.assert_allclose(
    residual,
    calculated_residual,
    rtol=0.0,
    atol=energy_atol,
)
np.testing.assert_allclose(
    radiation + cumulative_material + cumulative_boundary + cumulative_residual,
    radiation[0],
    rtol=0.0,
    atol=energy_atol,
)

# With diffusion disabled, changes in the thick field are solely LTE exchange.
# Remove that signed contribution to isolate the delayed streaming-work debit.
streaming_internal = internal + np.r_[0.0, np.diff(diffusion)]
negative_internal_rows = np.flatnonzero(streaming_internal < 0.0)
assert negative_internal_rows.size >= 1
assert np.all(negative_internal_rows > 0)
assert np.all(kinetic[negative_internal_rows] > 0.0)
assert np.all(
    kinetic[negative_internal_rows] > current_absorption[negative_internal_rows - 1]
)

applied_z = momentum["material_z(kg*m/s)"]
cumulative_applied_z = momentum["cumulative_material_z(kg*m/s)"]
pending_z = momentum["pending_streaming_material_z(kg*m/s)"]
requested_z = current_absorption / C_LIGHT
momentum_scale = max(
    float(np.max(np.abs(cumulative_applied_z))),
    float(np.max(np.abs(pending_z))),
    np.finfo(np.float64).tiny,
)
momentum_atol = 1024.0 * np.finfo(np.float64).eps * momentum_scale

assert np.any(pending_z != 0.0)
assert pending_z[-1] != 0.0
assert np.any(applied_z[1:] == 0.0)
assert np.count_nonzero(applied_z[1:]) >= 1
np.testing.assert_allclose(
    applied_z[1:] + np.diff(pending_z),
    requested_z,
    rtol=2.0e-10,
    atol=momentum_atol,
)
np.testing.assert_allclose(
    cumulative_applied_z,
    np.cumsum(applied_z),
    rtol=2.0e-11,
    atol=momentum_atol,
)
np.testing.assert_allclose(
    cumulative_applied_z[1:] + pending_z[1:],
    np.cumsum(requested_z),
    rtol=2.0e-10,
    atol=momentum_atol,
)

zero_momentum_labels = [
    "material_x(kg*m/s)",
    "material_y(kg*m/s)",
    "cumulative_material_x(kg*m/s)",
    "cumulative_material_y(kg*m/s)",
    "diffusion_boundary_x(kg*m/s)",
    "diffusion_boundary_y(kg*m/s)",
    "diffusion_boundary_z(kg*m/s)",
    "cumulative_diffusion_boundary_x(kg*m/s)",
    "cumulative_diffusion_boundary_y(kg*m/s)",
    "cumulative_diffusion_boundary_z(kg*m/s)",
    "streaming_boundary_x(kg*m/s)",
    "streaming_boundary_y(kg*m/s)",
    "streaming_boundary_z(kg*m/s)",
    "cumulative_streaming_boundary_x(kg*m/s)",
    "cumulative_streaming_boundary_y(kg*m/s)",
    "cumulative_streaming_boundary_z(kg*m/s)",
    "pending_streaming_material_x(kg*m/s)",
    "pending_streaming_material_y(kg*m/s)",
    "pending_diffusion_material_x(kg*m/s)",
    "pending_diffusion_material_y(kg*m/s)",
    "pending_diffusion_material_z(kg*m/s)",
]
for label in zero_momentum_labels:
    np.testing.assert_allclose(momentum[label], 0.0, rtol=0.0, atol=momentum_atol)

initial_weight, initial_particle_momentum = particle_state(Path("diags/diag000000"))
final_weight, final_particle_momentum = particle_state(Path("diags/diag000032"))
np.testing.assert_array_equal(final_weight, initial_weight)
particle_delta_momentum = np.sum(
    initial_weight[:, None] * (final_particle_momentum - initial_particle_momentum),
    axis=0,
)
np.testing.assert_allclose(
    particle_delta_momentum[2],
    cumulative_applied_z[-1],
    rtol=2.0e-10,
    atol=momentum_atol,
)
np.testing.assert_allclose(
    particle_delta_momentum[:2], 0.0, rtol=0.0, atol=momentum_atol
)

initial_u = initial_particle_momentum / ION_MASS_KG
final_u = final_particle_momentum / ION_MASS_KG
gamma_initial = np.sqrt(1.0 + np.sum(initial_u * initial_u, axis=1) / C_LIGHT**2)
gamma_final = np.sqrt(1.0 + np.sum(final_u * final_u, axis=1) / C_LIGHT**2)
particle_delta_per_ion = final_particle_momentum - initial_particle_momentum
particle_kinetic_change = np.sum(
    initial_weight
    * (
        2.0 * np.sum(initial_particle_momentum * particle_delta_per_ion, axis=1)
        + np.sum(particle_delta_per_ion * particle_delta_per_ion, axis=1)
    )
    / (ION_MASS_KG * (gamma_initial + gamma_final))
)
np.testing.assert_allclose(
    particle_kinetic_change,
    np.sum(kinetic),
    rtol=2.0e-10,
    atol=energy_atol,
)

physical_mass = float(initial_weight[0] * ION_MASS_KG)
initial_uz = float(initial_u[0, 2])
velocity_ulp = float(np.spacing(initial_uz))
np.testing.assert_allclose(initial_uz, 1.0e8, rtol=0.0, atol=velocity_ulp)
requested_delta_u = requested_z / physical_mass
assert np.all(requested_delta_u > 0.0)
assert np.max(requested_delta_u) < 0.1 * velocity_ulp
carry_bound = (
    256.0
    * np.finfo(np.float64).eps
    * physical_mass
    * max(float(np.linalg.norm(initial_u[0])), C_LIGHT)
)
assert float(np.max(np.abs(pending_z))) <= carry_bound

print(
    "streaming momentum carry: "
    f"max_pending={np.max(np.abs(pending_z)):.16e} kg*m/s, "
    f"negative_internal_steps={steps[negative_internal_rows].tolist()}, "
    f"particle_work={particle_kinetic_change:.16e} J"
)
