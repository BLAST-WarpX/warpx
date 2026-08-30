#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Conservative photon recoil test for an inward-moving RCYLINDER liner."""

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

assert radiation.shape == (2, 17)
assert momentum.shape == (2, 20)

initial_radiation = radiation[0, 2]
final_radiation = radiation[-1, 2]
absorbed_energy = initial_radiation - final_radiation
material_exchange = radiation[-1, 5]
internal_exchange = radiation[-1, 9]
kinetic_exchange = radiation[-1, 10]
radial_impulse = momentum[-1, 2]

# The packet stays in radial cell 8 of the 16-cell unit-radius mesh. The
# NUniformPerCell ion weights exactly represent this cell's physical mass per
# unit axial length.
n0 = 1.0e20
radius_lo = 8.0 / 16.0
radius_hi = 9.0 / 16.0
cell_mass = n0 * m_p * pi * (radius_hi**2 - radius_lo**2)
old_proper_velocity = -1.0e6
expected_impulse = absorbed_energy / c
proper_velocity_increment = expected_impulse / cell_mass
gamma_old = np.sqrt(1.0 + (old_proper_velocity / c) ** 2)
gamma_new = np.sqrt(1.0 + ((old_proper_velocity + proper_velocity_increment) / c) ** 2)
expected_kinetic_exchange = (
    cell_mass
    * (
        2.0 * old_proper_velocity * proper_velocity_increment
        + proper_velocity_increment**2
    )
    / (gamma_old + gamma_new)
)

print(f"absorbed radiation:      {absorbed_energy:.16e} J")
print(f"material radial impulse: {radial_impulse:.16e} kg*m/s")
print(f"ion kinetic change:      {kinetic_exchange:.16e} J")
print(f"electron internal gain:  {internal_exchange:.16e} J")

assert absorbed_energy > 0.0
assert expected_kinetic_exchange < 0.0
assert kinetic_exchange < 0.0, "radiation did not decelerate the inward liner"
assert internal_exchange > absorbed_energy
np.testing.assert_allclose(radial_impulse, expected_impulse, rtol=2.0e-12, atol=1.0e-18)
np.testing.assert_allclose(momentum[-1, 5], radial_impulse, rtol=2.0e-12, atol=1.0e-18)
np.testing.assert_allclose(
    kinetic_exchange,
    expected_kinetic_exchange,
    rtol=2.0e-12,
    atol=1.0e-9,
)
np.testing.assert_allclose(
    internal_exchange + kinetic_exchange,
    absorbed_energy,
    rtol=2.0e-12,
    atol=1.0e-9,
)
np.testing.assert_allclose(
    material_exchange, absorbed_energy, rtol=2.0e-12, atol=1.0e-9
)
np.testing.assert_allclose(
    momentum[:, [3, 4, 6, 7]],
    0.0,
    atol=1.0e-14 * abs(radial_impulse),
)
np.testing.assert_allclose(momentum[:, 8:20], 0.0, atol=1.0e-24)
assert_energy_balance(radiation_labels, radiation)
