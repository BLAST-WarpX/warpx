#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check metric hybrid LTE realization with a checkerboard source."""

import argparse
from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer
from scipy.constants import Boltzmann, c, elementary_charge, physical_constants


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


def compare_plotfiles(
    actual: dict[str, np.ndarray], reference: dict[str, np.ndarray], atol: float
) -> None:
    for name in FIELD_NAMES:
        np.testing.assert_allclose(actual[name], reference[name], rtol=0.0, atol=atol)


def physical_geometry(
    num_cells: int, dr: float, dz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radial_edges = np.arange(num_cells + 1, dtype=np.float64) * dr
    volumes = (
        np.pi * np.diff(radial_edges**2)[:, np.newaxis] * np.full((1, num_cells), dz)
    )
    corner_volumes = np.zeros((num_cells, num_cells, 2, 2))
    for i in range(num_cells):
        radial_mid = radial_edges[i] + 0.5 * dr
        radial_halves = (
            np.pi * (radial_mid**2 - radial_edges[i] ** 2),
            np.pi * (radial_edges[i + 1] ** 2 - radial_mid**2),
        )
        for j in range(num_cells):
            corner_volumes[i, j, :, :] = (
                0.5 * dz * np.asarray(radial_halves)[:, np.newaxis]
            )

    node_volumes = np.zeros((num_cells + 1, num_cells + 1))
    for i in range(num_cells + 1):
        radius = i * dr
        radial_lo = max(0.0, radius - 0.5 * dr)
        radial_hi = min(num_cells * dr, radius + 0.5 * dr)
        radial_volume = np.pi * (radial_hi**2 - radial_lo**2)
        for j in range(num_cells + 1):
            z_weight = 0.5 * dz if j in (0, num_cells) else dz
            node_volumes[i, j] = radial_volume * z_weight
    return radial_edges, volumes, corner_volumes, node_volumes


def reconstruct_temperature(
    source_cells: np.ndarray,
    old_temperature: np.ndarray,
    old_heat_capacity: np.ndarray,
    corner_volumes: np.ndarray,
    node_volumes: np.ndarray,
    equal_corners: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_cells = source_cells.shape[0]
    old_temperature_nodes = np.full(
        (num_cells + 1, num_cells + 1), np.mean(old_temperature)
    )
    old_heat_capacity_nodes = np.full(
        (num_cells + 1, num_cells + 1), np.mean(old_heat_capacity)
    )
    node_energy = np.zeros_like(node_volumes)
    for i in range(num_cells + 1):
        for j in range(num_cells + 1):
            for ci in (i - 1, i):
                for cj in (j - 1, j):
                    if not (0 <= ci < num_cells and 0 <= cj < num_cells):
                        continue
                    di = i - ci
                    dj = j - cj
                    if equal_corners:
                        node_energy[i, j] += 0.25 * source_cells[ci, cj]
                        continue
                    cell_capacity = 0.0
                    for corner_di in (0, 1):
                        for corner_dj in (0, 1):
                            cell_capacity += (
                                old_heat_capacity_nodes[ci + corner_di, cj + corner_dj]
                                * corner_volumes[ci, cj, corner_di, corner_dj]
                            )
                    node_capacity = (
                        old_heat_capacity_nodes[i, j] * corner_volumes[ci, cj, di, dj]
                    )
                    node_energy[i, j] += (
                        source_cells[ci, cj] * node_capacity / cell_capacity
                    )
    node_energy_density_change = node_energy / node_volumes
    node_temperature = old_temperature_nodes + (
        node_energy_density_change / old_heat_capacity_nodes
    )
    cell_temperature = 0.25 * (
        node_temperature[:-1, :-1]
        + node_temperature[1:, :-1]
        + node_temperature[:-1, 1:]
        + node_temperature[1:, 1:]
    )
    return node_temperature, cell_temperature, node_energy


def cell_realized_energy(
    node_temperature: np.ndarray,
    old_temperature: np.ndarray,
    old_heat_capacity: np.ndarray,
    corner_volumes: np.ndarray,
) -> np.ndarray:
    num_cells = old_temperature.shape[0]
    old_temperature_nodes = np.full(
        (num_cells + 1, num_cells + 1), np.mean(old_temperature)
    )
    old_heat_capacity_nodes = np.full(
        (num_cells + 1, num_cells + 1), np.mean(old_heat_capacity)
    )
    realized = np.zeros_like(old_temperature)
    for i in range(num_cells):
        for j in range(num_cells):
            for di in (0, 1):
                for dj in (0, 1):
                    realized[i, j] += (
                        old_heat_capacity_nodes[i + di, j + dj]
                        * (
                            node_temperature[i + di, j + dj]
                            - old_temperature_nodes[i + di, j + dj]
                        )
                        * corner_volumes[i, j, di, dj]
                    )
    return realized


parser = argparse.ArgumentParser()
parser.add_argument("--compare-reference", action="store_true")
parser.add_argument("--precision", choices=["SINGLE", "DOUBLE"], required=True)
args = parser.parse_args()

FIELD_NAMES = (
    "rho",
    "Te",
    "Pe",
    "radiation_material_energy",
    "radiation_diffusion_energy",
)
initial = load_plotfile(Path("diags/diag000001"))
final = load_plotfile(Path("diags/diag000002"))
radiation_table = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))

num_cells = 8
dr = 1.0e-3
dz = 1.0e-3
n0 = 2.857275673921765e19
raw_density = 0.25 * n0
minimum_radiation_density = 0.05 * n0
gamma = 5.0 / 3.0
temperature_0 = elementary_charge / Boltzmann
alpha_planck = 23.1209012123
dt = 1.0e-10
radiation_constant = 4.0 * physical_constants["Stefan-Boltzmann constant"][0] / c

if args.precision == "SINGLE":
    state_rtol = 5.0e-5
    ledger_rtol = 5.0e-6
    precision = np.float32
else:
    state_rtol = 3.0e-8
    ledger_rtol = 5.0e-10
    precision = np.float64
precision_eps = np.finfo(precision).eps

for fields in (initial, final):
    for name in FIELD_NAMES:
        assert fields[name].shape == (num_cells, num_cells)
        assert np.all(np.isfinite(fields[name]))

_, cell_volumes, corner_volumes, node_volumes = physical_geometry(num_cells, dr, dz)
np.testing.assert_allclose(
    np.sum(corner_volumes, axis=(2, 3)),
    cell_volumes,
    rtol=0.0,
    atol=64.0 * precision_eps * np.max(cell_volumes),
)
assert node_volumes[0, 0] > 0.0
assert node_volumes[-1, 0] > 0.0
assert node_volumes[0, 0] < node_volumes[0, 1]
assert node_volumes[0, -1] == node_volumes[0, 0]
assert node_volumes[-1, -1] == node_volumes[-1, 0]

radial_index = np.arange(num_cells)[:, np.newaxis]
axial_index = np.arange(num_cells)[np.newaxis, :]
active = (radial_index % 2) == (axial_index % 2)

rho_initial = initial["rho"]
rho_final = final["rho"]
temperature_initial = initial["Te"]
temperature_final = final["Te"]
pressure_initial = initial["Pe"]
pressure_final = final["Pe"]
radiation_initial = initial["radiation_diffusion_energy"]
radiation_final = final["radiation_diffusion_energy"]
material_initial = initial["radiation_material_energy"]
material_final = final["radiation_material_energy"]

number_density_initial = rho_initial / elementary_charge
number_density_final = rho_final / elementary_charge
for number_density in (number_density_initial, number_density_final):
    assert np.all(number_density > minimum_radiation_density)
    assert np.all(number_density < 0.5 * n0)
np.testing.assert_allclose(
    np.sum(number_density_initial * cell_volumes) / np.sum(cell_volumes),
    raw_density,
    rtol=0.2,
)
np.testing.assert_allclose(
    number_density_final, number_density_initial, rtol=state_rtol
)
np.testing.assert_allclose(temperature_initial, temperature_0, rtol=state_rtol)
np.testing.assert_allclose(
    pressure_initial / ((gamma - 1.0) * temperature_initial),
    n0 * Boltzmann / (gamma - 1.0),
    rtol=state_rtol,
)

exchange_fraction = -np.expm1(-alpha_planck * c * dt)
expected_radiation_density = exchange_fraction * radiation_constant * temperature_0**4
expected_radiation = np.zeros_like(cell_volumes)
expected_radiation[active] = expected_radiation_density * cell_volumes[active]
np.testing.assert_array_equal(radiation_initial, 0.0)
np.testing.assert_array_equal(material_initial, 0.0)
radiation_scale = max(np.max(np.abs(expected_radiation)), np.finfo(np.float64).tiny)
radiation_atol = 256.0 * precision_eps * radiation_scale
np.testing.assert_allclose(
    radiation_final, expected_radiation, rtol=state_rtol, atol=radiation_atol
)
np.testing.assert_array_equal(radiation_final[~active], 0.0)
assert np.all(radiation_final[active] > 0.0)

# q is the measured radiation-to-material request.  The analytic source check
# above remains independent of this residual oracle.
requested_cells = radiation_initial - radiation_final
q = np.sum(requested_cells)
assert q < 0.0
np.testing.assert_allclose(
    q, -np.sum(expected_radiation), rtol=state_rtol, atol=radiation_atol
)

energy_density_initial = pressure_initial / (gamma - 1.0)
heat_capacity_initial = energy_density_initial / temperature_initial
node_temperature, _, _ = reconstruct_temperature(
    requested_cells,
    temperature_initial,
    heat_capacity_initial,
    corner_volumes,
    node_volumes,
)
mutant_node_temperature, _, _ = reconstruct_temperature(
    requested_cells,
    temperature_initial,
    heat_capacity_initial,
    corner_volumes,
    node_volumes,
    equal_corners=True,
)
assert np.all(temperature_final > 0.0)
assert np.all(temperature_final < temperature_0)
assert np.all(pressure_final > 0.0)
assert np.all(pressure_final < pressure_initial)

expected_material_cells = cell_realized_energy(
    node_temperature,
    temperature_initial,
    heat_capacity_initial,
    corner_volumes,
)
material_change = material_final - material_initial
energy_scale = max(
    np.max(np.abs(material_change)),
    np.max(np.abs(expected_material_cells)),
    abs(q),
    np.finfo(np.float64).tiny,
)
energy_atol = 256.0 * precision_eps * energy_scale
np.testing.assert_allclose(
    material_change, expected_material_cells, rtol=ledger_rtol, atol=energy_atol
)
d = np.sum(material_change)
assert np.any(np.abs(material_change[~active]) > 64.0 * precision_eps * energy_scale)
np.testing.assert_array_equal(requested_cells[~active], 0.0)
np.testing.assert_allclose(
    material_change[~active],
    expected_material_cells[~active],
    rtol=ledger_rtol,
    atol=energy_atol,
)
boundary_energy_threshold = 64.0 * precision_eps * energy_scale
for boundary_name, boundary_values in (
    ("axis", expected_material_cells[0, :]),
    ("outer radius", expected_material_cells[-1, :]),
    ("lower z", expected_material_cells[:, 0]),
    ("upper z", expected_material_cells[:, -1]),
):
    assert np.any(np.abs(boundary_values) > boundary_energy_threshold), boundary_name

mutant_material = cell_realized_energy(
    mutant_node_temperature,
    temperature_initial,
    heat_capacity_initial,
    corner_volumes,
)
mutant_gap = np.max(np.abs(expected_material_cells - mutant_material))
assert mutant_gap > max(1.0e-6 * energy_scale, 64.0 * precision_eps * energy_scale)

assert radiation_table.shape[0] == 3
assert radiation_table.shape[1] >= 17
unrelated_columns = np.array([3, 7, 8, 10, 11, 12, 13, 14])
np.testing.assert_allclose(
    radiation_table[:, unrelated_columns], 0.0, rtol=0.0, atol=energy_atol
)
np.testing.assert_allclose(radiation_table[:2, -2:], 0.0, rtol=0.0, atol=energy_atol)
residual = radiation_table[-1, -2]
cumulative_residual = radiation_table[-1, -1]
np.testing.assert_allclose(residual, q - d, rtol=0.0, atol=energy_atol)
np.testing.assert_allclose(cumulative_residual, residual, rtol=0.0, atol=energy_atol)
np.testing.assert_allclose(
    radiation_table[0, 2] - radiation_table[-1, 2], q, rtol=0.0, atol=energy_atol
)
np.testing.assert_allclose(
    radiation_table[-1, 4], np.sum(radiation_final), rtol=ledger_rtol, atol=energy_atol
)
np.testing.assert_allclose(radiation_table[-1, 5], d, rtol=0.0, atol=energy_atol)
np.testing.assert_allclose(radiation_table[-1, 6], d, rtol=0.0, atol=energy_atol)
np.testing.assert_allclose(radiation_table[-1, 9], d, rtol=0.0, atol=energy_atol)

if args.compare_reference:
    reference_dir = Path.cwd().with_name(Path.cwd().name.removesuffix("_mpi"))
    reference_initial = load_plotfile(reference_dir / "diags/diag000001")
    reference_final = load_plotfile(reference_dir / "diags/diag000002")
    reference_table = np.atleast_2d(
        np.loadtxt(reference_dir / "diags/radiation_energy.txt")
    )
    compare_scale = max(
        np.max(np.abs(radiation_table)),
        np.max(np.abs(reference_table)),
        *(np.max(np.abs(initial[name])) for name in FIELD_NAMES),
        *(np.max(np.abs(reference_initial[name])) for name in FIELD_NAMES),
        *(np.max(np.abs(final[name])) for name in FIELD_NAMES),
        *(np.max(np.abs(reference_final[name])) for name in FIELD_NAMES),
        np.finfo(np.float64).tiny,
    )
    compare_atol = 256.0 * precision_eps * compare_scale
    compare_plotfiles(initial, reference_initial, compare_atol)
    compare_plotfiles(final, reference_final, compare_atol)
    np.testing.assert_allclose(
        radiation_table, reference_table, rtol=0.0, atol=compare_atol
    )

print(
    "RZ hybrid near-floor realization: "
    f"Q={-q:.16e} J, q={q:.16e} J, d={d:.16e} J, "
    f"R={residual:.16e} J, cumulative_R={cumulative_residual:.16e} J, "
    f"zero-q d={np.max(np.abs(material_change[~active])):.16e} J, "
    f"equal-corner kill gap={mutant_gap:.16e} J"
)
