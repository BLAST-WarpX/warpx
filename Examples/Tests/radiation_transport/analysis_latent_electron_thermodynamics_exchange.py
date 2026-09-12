#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Validate nonlinear latent-electron LTE exchange and RCYL energy closure."""

import argparse
from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer
from scipy.constants import Boltzmann, c, elementary_charge


def load_plotfile(path: Path) -> dict[str, np.ndarray]:
    with open(path / "Header") as header:
        header.readline()
        n_fields = int(header.readline())
        field_names = [header.readline().strip() for _ in range(n_fields)]
    fields = _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), field_names)
    return {
        name: np.asarray(value, dtype=np.float64).squeeze()
        for name, value in fields.items()
    }


parser = argparse.ArgumentParser()
parser.add_argument("--precision", choices=["SINGLE", "DOUBLE"], default="DOUBLE")
args = parser.parse_args()

if args.precision == "SINGLE":
    state_rtol = 5.0e-6
    analytic_rtol = 1.0e-5
    ledger_rtol = 1.0e-5
else:
    state_rtol = 5.0e-11
    analytic_rtol = 5.0e-9
    ledger_rtol = 5.0e-10

# The nonlinear U_e scratch field is nodal and the plotfile stores its
# arithmetic cell image.  Integrating that image with annular cell volumes is
# not the exact cylindrical dual-volume quadrature used by the material
# ledger.  For this fixed 31-ring, strongly nonuniform-temperature mesh the
# resulting geometry error is about 1.92e-4.  A 5e-4 diagnostic-only allowance
# leaves portability margin while remaining 1500 times below the 0.75 relative
# error produced by evaluating the adapter at n_floor instead of raw rho.
diagnostic_metric_rtol = 5.0e-4

# Iteration 0 precedes initialization.  Step 1 follows a zero-opacity warm-up;
# step 2 follows the single active, frozen-temperature LTE exchange.
initial = load_plotfile(Path("diags/diag000001"))
final = load_plotfile(Path("diags/diag000002"))
radiation_table = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))

n_cell = 31
radius = 1.0e-2
edges = np.linspace(0.0, radius, n_cell + 1)
volumes = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
emitting = np.arange(n_cell) % 2 == 0

for fields in (initial, final):
    for name in (
        "rho",
        "Te",
        "Pe",
        "hybrid_qdsmc_thermodynamic_fp",
        "radiation_material_energy",
        "radiation_diffusion_energy",
    ):
        assert fields[name].shape == (n_cell,)

gamma = 5.0 / 3.0
n_floor = 1.0e25
minimum_radiation_density = 0.05 * n_floor
temperature_0_eV = 100.0
transition_temperature_eV = 100.0
latent_energy_eV = 1000.0
latent_sharpness = 4.0
temperature_0 = temperature_0_eV * elementary_charge / Boltzmann
alpha_planck = 5.0e2
dt = 4.0e-13
radiation_constant = 7.565733250280007e-16


def occupation(temperature_eV):
    scaled_temperature = np.asarray(temperature_eV) / transition_temperature_eV
    return scaled_temperature**latent_sharpness / (
        1.0 + scaled_temperature**latent_sharpness
    )


def energy_per_electron_eV(temperature_eV):
    return temperature_eV / (gamma - 1.0) + latent_energy_eV * occupation(
        temperature_eV
    )


rho_0 = initial["rho"]
rho_1 = final["rho"]
ne_0 = rho_0 / elementary_charge
ne_1 = rho_1 / elementary_charge
temperature_initial = initial["Te"]
temperature_final = final["Te"]
pressure_initial = initial["Pe"]
pressure_final = final["Pe"]
internal_energy_initial = initial["hybrid_qdsmc_thermodynamic_fp"]
internal_energy_final = final["hybrid_qdsmc_thermodynamic_fp"]
radiation_initial = initial["radiation_diffusion_energy"]
radiation_final = final["radiation_diffusion_energy"]
material_initial = initial["radiation_material_energy"]
material_final = final["radiation_material_energy"]

# The nonlinear finite-volume state is defined at the raw deposited density;
# n_floor is only the Ohm-law denominator floor.  The radiation producer,
# nonlinear inverse, realized material ledger, QDSMC U_e, and emitted pressure
# must all use this same raw-density EOS state.
assert np.min(ne_0[emitting]) > minimum_radiation_density
assert np.max(ne_0) < 0.5 * n_floor
assert 0.20 * n_floor < np.mean(ne_0[emitting]) < 0.30 * n_floor
np.testing.assert_allclose(ne_1, ne_0, rtol=state_rtol)
np.testing.assert_allclose(temperature_initial, temperature_0, rtol=state_rtol)
# The required reflecting outer particle boundary applies WarpX's pressure
# boundary transform to the last plotted Pe and U_e cells.  The interior cells
# remain the direct raw-rho EOS image; the complete U_e change below is still
# the authoritative transported-energy oracle, including that boundary state.
pressure_eos_cells = slice(0, -1)
np.testing.assert_allclose(
    pressure_initial[pressure_eos_cells],
    ne_0[pressure_eos_cells] * Boltzmann * temperature_0,
    rtol=state_rtol,
)
np.testing.assert_allclose(
    internal_energy_initial[pressure_eos_cells],
    ne_0[pressure_eos_cells]
    * elementary_charge
    * energy_per_electron_eV(temperature_0_eV),
    rtol=state_rtol,
)

# Frozen-temperature LTE emission is an exact Beer-Lambert relaxation toward
# a*T0^4.  The manufactured per-cell opacity accounts for the exact RCYL dual
# volumes so every node loses the same energy density after the inverse remap.
exchange_fraction = -np.expm1(-alpha_planck * c * dt)
radiation_equilibrium_density = radiation_constant * temperature_0**4
node_energy_density = exchange_fraction * radiation_equilibrium_density

dr = radius / n_cell
node_radii = edges
node_lo = np.maximum(0.0, node_radii - 0.5 * dr)
node_hi = np.minimum(radius, node_radii + 0.5 * dr)
node_volumes = np.pi * (node_hi**2 - node_lo**2)
source_volume_ratio = (node_volumes[:-1] + node_volumes[1:]) / volumes
expected_radiation = np.zeros(n_cell)
expected_radiation[emitting] = (
    node_energy_density * source_volume_ratio[emitting] * volumes[emitting]
)
expected_total = np.sum(expected_radiation)
np.testing.assert_allclose(
    expected_total, node_energy_density * np.sum(node_volumes), rtol=1.0e-15
)

np.testing.assert_array_equal(radiation_initial, 0.0)
np.testing.assert_array_equal(material_initial, 0.0)
np.testing.assert_allclose(
    radiation_final[emitting], expected_radiation[emitting], rtol=analytic_rtol
)
np.testing.assert_array_equal(radiation_final[~emitting], 0.0)

# Raw PIC density is deliberately nonuniform, so a fixed cell energy request
# produces a physical nonuniform temperature response.  Every state remains
# finite, positive, and cooler, while the directly exposed nonlinear U_e field
# supplies the independent electron-energy oracle.
for field in (temperature_final, pressure_final, internal_energy_final):
    assert np.all(np.isfinite(field))
    assert np.all(field > 0.0)
assert np.all(temperature_final < temperature_initial)
assert np.all(pressure_final < pressure_initial)
assert np.all(internal_energy_final < internal_energy_initial)
assert np.ptp(temperature_final) > 0.1 * np.mean(temperature_final)

material_change = material_final - material_initial
assert np.all(material_change < 0.0)
# Shared nodes spread the realized EOS change into cells with q=0.  Retain a
# strong spatial check so a cell-local requested-energy ledger cannot pass.
assert np.max(np.abs(material_change[~emitting])) > 1.0e-3 * expected_total
realized_material = np.sum(material_change)
electron_change_cc = np.sum((internal_energy_final - internal_energy_initial) * volumes)
electron_ledger_relative_gap = abs(electron_change_cc - realized_material) / max(
    abs(electron_change_cc), abs(realized_material)
)
assert electron_ledger_relative_gap < diagnostic_metric_rtol

# Pe/(gamma-1) is only the ideal translational part.  The directly diagnosed
# nonlinear U_e change must be substantially larger because the latent
# reservoir is active; otherwise this fixture has silently become ideal gas.
ideal_pressure_change = np.sum(
    (pressure_final[pressure_eos_cells] - pressure_initial[pressure_eos_cells])
    * volumes[pressure_eos_cells]
    / (gamma - 1.0)
)
nonlinear_energy_change_on_pressure_cells = np.sum(
    (
        internal_energy_final[pressure_eos_cells]
        - internal_energy_initial[pressure_eos_cells]
    )
    * volumes[pressure_eos_cells]
)
latent_to_ideal_ratio = abs(
    nonlinear_energy_change_on_pressure_cells / ideal_pressure_change
)
assert latent_to_ideal_ratio > 2.0

# The reduced diagnostic supplies an independent radiation total and cumulative
# material ledger.  The stationary dummy photon and closed diffusion state have
# no other energy or boundary channels.  The requested transfer is
# q=-Delta E_radiation, while the material field and all three material ledgers
# record the EOS-realized d.  Their difference is the numerical residual R=q-d.
assert radiation_table.shape[0] == 3
assert radiation_table.shape[1] >= 17
np.testing.assert_allclose(radiation_table[-1, 4], expected_total, rtol=analytic_rtol)
np.testing.assert_allclose(
    radiation_table[-1, 4], np.sum(radiation_final), rtol=ledger_rtol
)
np.testing.assert_allclose(radiation_table[-1, 5], realized_material, rtol=ledger_rtol)
np.testing.assert_allclose(radiation_table[-1, 6], realized_material, rtol=ledger_rtol)
np.testing.assert_allclose(radiation_table[-1, 9], realized_material, rtol=ledger_rtol)
np.testing.assert_allclose(
    radiation_table[:, 3], radiation_table[1, 3], rtol=ledger_rtol
)
np.testing.assert_allclose(radiation_table[:, 7:9], 0.0, atol=1.0e-24)
np.testing.assert_allclose(radiation_table[:, 10:-2], 0.0, atol=1.0e-24)

# q is the measured radiation-to-material transfer.  Keep the analytic LTE
# result above as an independent radiation check; do not let its solver/float
# error contaminate the q-d residual oracle.
requested_material = np.sum(radiation_initial) - np.sum(radiation_final)
residual = radiation_table[-1, -2]
cumulative_residual = radiation_table[-1, -1]
precision = np.float32 if args.precision == "SINGLE" else np.float64
field_roundoff = (
    32.0
    * np.finfo(precision).eps
    * max(
        abs(radiation_table[0, 2]),
        abs(radiation_table[-1, 2]),
        abs(requested_material),
        abs(realized_material),
        abs(radiation_table[-1, 5]),
        abs(radiation_table[-1, 6]),
        abs(radiation_table[-1, 9]),
    )
)
field_tolerance = max(1.0e-24, field_roundoff)
closure_residual = radiation_table[0, 2] - radiation_table[-1, 2] - realized_material
np.testing.assert_allclose(residual, closure_residual, rtol=0.0, atol=field_tolerance)
np.testing.assert_allclose(
    residual,
    requested_material - realized_material,
    rtol=0.0,
    atol=field_tolerance,
)
np.testing.assert_allclose(
    cumulative_residual, residual, rtol=0.0, atol=field_tolerance
)
np.testing.assert_allclose(
    radiation_table[:2, -2:], 0.0, rtol=0.0, atol=field_tolerance
)
np.testing.assert_allclose(
    realized_material + radiation_table[-1, 2] - radiation_table[0, 2],
    -residual,
    rtol=0.0,
    atol=field_tolerance,
)

print(
    "RCYL latent-electron LTE exchange: "
    f"raw_ne/n_floor={np.mean(ne_0[emitting]) / n_floor:.8f}, "
    f"T1=[{np.min(temperature_final) * Boltzmann / elementary_charge:.6f}, "
    f"{np.max(temperature_final) * Boltzmann / elementary_charge:.6f}] eV, "
    f"electron/ledger gap={electron_ledger_relative_gap:.6e}, "
    f"latent/ideal={latent_to_ideal_ratio:.6f}, "
    f"radiation={expected_total:.10e} J, "
    f"realized material={realized_material:.10e} J, "
    f"numerical residual={residual:.6e} J"
)
