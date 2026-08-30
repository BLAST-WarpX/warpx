#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Deterministic two-group FLD radiation force and work test in RCYLINDER."""

from pathlib import Path

import numpy as np
from scipy.constants import c, m_p, pi


def load_table(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open() as stream:
        labels = [
            token.split("]", 1)[1]
            for token in stream.readline().lstrip("#").split()
            if token.startswith("[") and "]" in token
        ]
    data = np.atleast_2d(np.loadtxt(path))
    assert len(labels) == data.shape[1]
    return labels, data


def assert_energy_balance(labels: list[str], data: np.ndarray) -> None:
    columns = {name: labels.index(name) for name in labels}
    total = data[:, columns["total_radiation(J)"]]
    material = data[:, columns["material_exchange(J)"]]
    cumulative_material = data[:, columns["cumulative_material_exchange(J)"]]
    boundary = data[:, columns["boundary_energy_loss(J)"]]
    cumulative_boundary = data[:, columns["cumulative_boundary_energy_loss(J)"]]
    residual = data[:, columns["numerical_energy_residual(J)"]]
    cumulative_residual = data[:, columns["cumulative_numerical_energy_residual(J)"]]

    scale = max(
        1.0,
        float(
            np.max(
                np.abs(
                    data[
                        :,
                        [
                            columns["total_radiation(J)"],
                            columns["material_exchange(J)"],
                            columns["cumulative_material_exchange(J)"],
                            columns["boundary_energy_loss(J)"],
                            columns["cumulative_boundary_energy_loss(J)"],
                        ],
                    ]
                )
            )
        ),
    )
    field_atol = max(1.0e-24, 32.0 * np.finfo(np.float64).eps * scale)
    np.testing.assert_allclose(residual[0], 0.0, rtol=0.0, atol=field_atol)
    np.testing.assert_allclose(cumulative_residual[0], 0.0, rtol=0.0, atol=field_atol)

    current_balance = total[:-1] - total[1:] - material[1:] - boundary[1:]
    cumulative_balance = (
        total[0] - total[1:] - cumulative_material[1:] - cumulative_boundary[1:]
    )
    np.testing.assert_allclose(residual[1:], current_balance, rtol=0.0, atol=field_atol)
    np.testing.assert_allclose(
        cumulative_residual[1:], cumulative_balance, rtol=0.0, atol=field_atol
    )
    np.testing.assert_allclose(
        cumulative_residual[1:],
        np.cumsum(residual[1:]),
        rtol=0.0,
        atol=field_atol,
    )
    np.testing.assert_allclose(
        total[-1]
        + cumulative_material[-1]
        + cumulative_boundary[-1]
        + cumulative_residual[-1],
        total[0],
        rtol=0.0,
        atol=field_atol,
    )

    boundary_columns = [
        columns["streaming_boundary_energy_loss(J)"],
        columns["cumulative_streaming_boundary_energy_loss(J)"],
        columns["diffusion_boundary_energy_loss(J)"],
        columns["cumulative_diffusion_boundary_energy_loss(J)"],
    ]
    np.testing.assert_allclose(data[:, boundary_columns], 0.0, atol=1.0e-18)


radiation_labels, radiation = load_table(Path("diags/radiation_energy.txt"))
_, momentum = load_table(Path("diags/radiation_momentum.txt"))

assert radiation.shape == (2, 19)
assert momentum.shape == (2, 20)

dt = 1.0e-12
radius = 0.02
num_cells = 4
dr = radius / num_cells
rosseland = 400.0
n0 = 1.0e20

# The localized LTE source occupies cell 1. The material ledger gives the
# realized, energy-limited emission without relying on the source heat-capacity
# remap. Equal transport coefficients make the summed group force exactly the
# grey force evaluated from the total emitted energy.
emitted = -radiation[-1, 11]
source_volume = 3.0 * pi * dr**2
energy_density = emitted / source_volume
dimensionless_gradient = 2.0 / (rosseland * dr)
flux_limiter = (2.0 + dimensionless_gradient) / (
    6.0 + 3.0 * dimensionless_gradient + dimensionless_gradient**2
)
face_flux = c * flux_limiter * energy_density / (rosseland * dr)

volume_0 = pi * dr**2
volume_2 = 5.0 * pi * dr**2
impulse_0 = -dt * volume_0 * rosseland * face_flux / (2.0 * c)
impulse_2 = dt * volume_2 * rosseland * face_flux / (2.0 * c)
expected_impulse = impulse_0 + impulse_2

mass_0 = n0 * m_p * volume_0
mass_2 = n0 * m_p * volume_2
proper_velocity_0 = impulse_0 / mass_0
proper_velocity_2 = impulse_2 / mass_2
kinetic_0 = (
    mass_0 * proper_velocity_0**2 / (1.0 + np.sqrt(1.0 + (proper_velocity_0 / c) ** 2))
)
kinetic_2 = (
    mass_2 * proper_velocity_2**2 / (1.0 + np.sqrt(1.0 + (proper_velocity_2 / c) ** 2))
)
expected_kinetic = kinetic_0 + kinetic_2

group_energies = radiation[-1, 9:11]
diffusion_energy = radiation[-1, 4]
material_impulse = momentum[-1, 2]
material_kinetic = radiation[-1, 12]
group_fractions = group_energies / np.sum(group_energies)

print(f"emitted diffusion energy: {emitted:.16e} J")
print(f"surviving group energies: {group_energies}")
print(f"surviving group fractions: {group_fractions}")
print(f"material radial impulse:  {material_impulse:.16e} kg*m/s")
print(f"material kinetic work:    {material_kinetic:.16e} J")

assert emitted > 0.0
assert material_kinetic > 0.0
assert np.all(group_energies > 0.0)
assert np.all((group_fractions > 0.45) & (group_fractions < 0.55))
np.testing.assert_allclose(
    np.sum(group_energies), diffusion_energy, rtol=2.0e-12, atol=1.0e-9
)
np.testing.assert_allclose(
    material_impulse, expected_impulse, rtol=2.0e-12, atol=1.0e-20
)
np.testing.assert_allclose(
    momentum[-1, 5], material_impulse, rtol=2.0e-12, atol=1.0e-20
)
np.testing.assert_allclose(
    material_kinetic, expected_kinetic, rtol=2.0e-10, atol=1.0e-12
)
np.testing.assert_allclose(
    diffusion_energy + material_kinetic,
    emitted,
    rtol=2.0e-12,
    atol=1.0e-9,
)
np.testing.assert_allclose(
    radiation[-1, 5], -emitted + material_kinetic, rtol=2.0e-12, atol=1.0e-9
)
np.testing.assert_allclose(
    radiation[-1, 6], radiation[-1, 5], rtol=2.0e-12, atol=1.0e-9
)

transverse_tolerance = max(1.0e-20, 1.0e-14 * abs(material_impulse))
np.testing.assert_allclose(momentum[:, [3, 4, 6, 7]], 0.0, atol=transverse_tolerance)
np.testing.assert_allclose(momentum[:, 8:20], 0.0, atol=1.0e-24)
assert_energy_balance(radiation_labels, radiation)
