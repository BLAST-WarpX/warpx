#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Validate species-aware grey LTE emission in RCYLINDER."""

from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer
from scipy.constants import elementary_charge


def load_plotfile(path: Path) -> dict[str, np.ndarray]:
    with open(path / "Header") as header:
        header.readline()
        n_fields = int(header.readline())
        field_names = [header.readline().strip() for _ in range(n_fields)]
    return _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), field_names)


def load_table(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open() as stream:
        labels = [
            token.split("]", 1)[1]
            for token in stream.readline().lstrip("#").split()
            if token.startswith("[") and "]" in token
        ]
    table = np.atleast_2d(np.loadtxt(path))
    assert len(labels) == table.shape[1]
    return labels, table


initial = load_plotfile(Path("diags/diag1000000"))
final = load_plotfile(Path("diags/diag1000001"))
radiation_labels, radiation_table = load_table(Path("diags/radiation_energy.txt"))

radiation = final["radiation_diffusion_energy"].squeeze()
initial_radiation = initial["radiation_diffusion_energy"].squeeze()
material_exchange = final["radiation_material_energy"].squeeze()
initial_density = initial["rho"].squeeze() / elementary_charge
initial_temperature = initial["Te"].squeeze()
final_temperature = final["Te"].squeeze()

num_cells = 12
radius = 12.0e-3
dr = radius / num_cells

cell_index = np.arange(num_cells)
cell_center = (cell_index + 0.5) * dr
foam = (cell_center >= 2.0 * dr) & (cell_center < 4.0 * dr)
tungsten = (cell_center >= 6.0 * dr) & (cell_center < 8.0 * dr)
background = ~(foam | tungsten)

assert radiation.shape == (num_cells,)
assert radiation_table.shape[0] == 2
assert radiation_table.shape[1] >= 10
single_precision = radiation.dtype == np.float32
analytic_rtol = 2.0e-5 if single_precision else 5.0e-12
precision = np.dtype(np.float32 if single_precision else np.float64)
columns = {name: radiation_labels.index(name) for name in radiation_labels}
radiation_active = np.asarray(radiation, dtype=precision)
material_exchange_active = np.asarray(material_exchange, dtype=precision)
radiation_table_active = np.asarray(radiation_table, dtype=precision)

# The unlisted background supplies electrons in every cell. Its exact zero
# radiation is therefore a species-selection check, not an electron-density
# gate. The two named annuli must be the complete support of LTE emission.
assert np.all(initial_density > 0.0)
assert np.all(initial_temperature > 0.0)
np.testing.assert_array_equal(initial_radiation, 0.0)
np.testing.assert_array_equal(radiation[background], 0.0)
assert np.all(radiation[foam] > 0.0)
assert np.all(radiation[tungsten] > 0.0)
source_indices = np.flatnonzero(radiation_active)
np.testing.assert_array_equal(source_indices, np.flatnonzero(~background))

# RCYL cell-to-node realization spreads each source over the source cell and
# its immediate neighbors. Require that support exactly, including a negative
# material-energy halo and exact zero outside it.
material_support = np.zeros(num_cells, dtype=bool)
for index in source_indices:
    material_support[max(index - 1, 0) : min(index + 2, num_cells)] = True
np.testing.assert_array_equal(
    np.flatnonzero(material_exchange), np.flatnonzero(material_support)
)
assert np.all(material_exchange_active[material_support] < 0.0)
np.testing.assert_array_equal(material_exchange_active[~material_support], 0.0)

# The source is applied to the public ideal-electron closure, so every emitting
# cell must cool. Its nodal RCYL state is intentionally not reconstructed from
# the cell-averaged plot field; the exact conservative check is the material
# ledger below, while generic hybrid-radiation tests cover the inverse map.
assert np.all(final_temperature[foam | tungsten] < initial_temperature[foam | tungsten])
diffusion_gain = np.sum(radiation_active, dtype=precision)

# Independently require the reduced-diagnostic material+radiation invariant,
# including the signed current and cumulative numerical residuals.
total = radiation_table_active[:, columns["total_radiation(J)"]]
material_column = columns["material_exchange(J)"]
cumulative_material_column = columns["cumulative_material_exchange(J)"]
boundary_column = columns["boundary_energy_loss(J)"]
cumulative_boundary_column = columns["cumulative_boundary_energy_loss(J)"]
residual_column = columns["numerical_energy_residual(J)"]
cumulative_residual_column = columns["cumulative_numerical_energy_residual(J)"]
material_sum = np.sum(material_exchange_active, dtype=precision)
energy_scale = max(
    float(
        np.max(
            np.abs(
                radiation_table_active[
                    :,
                    [
                        columns["total_radiation(J)"],
                        material_column,
                        cumulative_material_column,
                        boundary_column,
                        cumulative_boundary_column,
                    ],
                ]
            )
        )
    ),
    abs(material_sum),
)
field_tolerance = max(
    np.finfo(precision).tiny, 32.0 * np.finfo(precision).eps * energy_scale
)
np.testing.assert_allclose(
    radiation_table_active[:, boundary_column], 0.0, rtol=0.0, atol=field_tolerance
)
np.testing.assert_allclose(
    radiation_table_active[-1, columns["diffusion_radiation(J)"]],
    diffusion_gain,
    rtol=analytic_rtol,
)
np.testing.assert_allclose(
    radiation_table_active[-1, material_column],
    material_sum,
    rtol=0.0,
    atol=field_tolerance,
)
np.testing.assert_allclose(
    precision.type(
        radiation_table_active[-1, cumulative_material_column]
        - radiation_table_active[0, cumulative_material_column]
    ),
    material_sum,
    rtol=0.0,
    atol=field_tolerance,
)
np.testing.assert_allclose(
    radiation_table_active[-1, columns["material_internal_exchange(J)"]],
    material_sum,
    rtol=0.0,
    atol=field_tolerance,
)
current_residual = radiation_table_active[-1, residual_column]
cumulative_material_increment = precision.type(
    radiation_table_active[-1, cumulative_material_column]
    - radiation_table_active[0, cumulative_material_column]
)
cumulative_boundary_increment = precision.type(
    radiation_table_active[-1, cumulative_boundary_column]
    - radiation_table_active[0, cumulative_boundary_column]
)
cumulative_residual_increment = precision.type(
    radiation_table_active[-1, cumulative_residual_column]
    - radiation_table_active[0, cumulative_residual_column]
)
requested_material = precision.type(total[0] - total[-1])
expected_current_residual = precision.type(
    precision.type(requested_material - radiation_table_active[-1, material_column])
    - radiation_table_active[-1, boundary_column]
)
np.testing.assert_allclose(
    current_residual,
    expected_current_residual,
    rtol=0.0,
    atol=field_tolerance,
)
expected_cumulative_residual = precision.type(
    precision.type(precision.type(total[0] - total[-1]) - cumulative_material_increment)
    - cumulative_boundary_increment
)
np.testing.assert_allclose(
    cumulative_residual_increment,
    expected_cumulative_residual,
    rtol=0.0,
    atol=field_tolerance,
)
np.testing.assert_allclose(
    cumulative_residual_increment,
    current_residual,
    rtol=0.0,
    atol=field_tolerance,
)
balance_residual = precision.type(total[-1] - total[0])
balance_residual = precision.type(balance_residual + cumulative_material_increment)
balance_residual = precision.type(balance_residual + cumulative_boundary_increment)
balance_residual = precision.type(balance_residual + cumulative_residual_increment)
np.testing.assert_allclose(balance_residual, 0.0, rtol=0.0, atol=field_tolerance)

print(
    "RCYL species opacity: "
    f"foam={np.sum(radiation[foam]):.16e} J, "
    f"tungsten={np.sum(radiation[tungsten]):.16e} J, "
    f"material+radiation residual={balance_residual:.6e} J"
)
