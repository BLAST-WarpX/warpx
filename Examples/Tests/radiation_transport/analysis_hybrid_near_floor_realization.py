#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check representable radiation with below-ULP EOS material realization."""

import argparse
from pathlib import Path

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from read_raw_data import _read_buffer
from scipy.constants import Boltzmann, c, elementary_charge


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


parser = argparse.ArgumentParser()
parser.add_argument("--compare-reference", action="store_true")
add_precision_arguments(parser)
args = parser.parse_args()
field_dtype, _, cross_dtype = precision_dtypes(args)

initial = load_plotfile(Path("diags/diag000001"))
final = load_plotfile(Path("diags/diag000002"))
radiation_table = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))

num_cells = 8
dx = 1.0
n0 = 1.0e20
raw_density = 0.25 * n0
minimum_radiation_density = 0.05 * n0
gamma = 5.0 / 3.0
temperature_0_eV = 1.0e-3
temperature_0 = temperature_0_eV * elementary_charge / Boltzmann
alpha_planck = 1.9881969996639316e-6
dt = 1.0e-10
radiation_constant = 7.565733250280007e-16

if field_dtype == np.float32:
    state_rtol = 5.0e-5
    field_eps = np.finfo(np.float32).eps
else:
    state_rtol = 3.0e-12
    field_eps = np.finfo(np.float64).eps
if cross_dtype == np.float32:
    cross_state_rtol = 5.0e-5
    cross_analytic_rtol = 5.0e-5
    ledger_rtol = 5.0e-5
else:
    cross_state_rtol = 3.0e-12
    cross_analytic_rtol = 3.0e-12
    ledger_rtol = 3.0e-12
residual_multiplier = 256.0
cross_eps = np.finfo(cross_dtype).eps

field_names = (
    "rho",
    "Te",
    "Pe",
    "radiation_material_energy",
    "radiation_diffusion_energy",
)
for fields in (initial, final):
    for name in field_names:
        assert fields[name].shape == (num_cells,)
        assert np.all(np.isfinite(fields[name]))

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

state_scale = max(
    np.max(np.abs(rho_initial)),
    np.max(np.abs(rho_final)),
    np.max(np.abs(temperature_initial)),
    np.max(np.abs(temperature_final)),
    np.max(np.abs(pressure_initial)),
    np.max(np.abs(pressure_final)),
)
state_atol = residual_multiplier * field_eps * state_scale
np.testing.assert_allclose(rho_initial / elementary_charge, raw_density, rtol=0.2)
assert np.all(rho_initial / elementary_charge > minimum_radiation_density)
assert np.all(rho_initial / elementary_charge < 0.5 * n0)
np.testing.assert_allclose(rho_final, rho_initial, rtol=state_rtol, atol=state_atol)
np.testing.assert_allclose(
    temperature_initial,
    temperature_0,
    rtol=cross_state_rtol,
    atol=max(state_atol, residual_multiplier * cross_eps * state_scale),
)
np.testing.assert_allclose(
    temperature_final, temperature_initial, rtol=0.0, atol=state_atol
)
np.testing.assert_allclose(pressure_final, pressure_initial, rtol=0.0, atol=state_atol)

exchange_fraction = -np.expm1(-alpha_planck * c * dt)
expected_radiation_density = exchange_fraction * radiation_constant * temperature_0**4
expected_radiation = np.full(num_cells, expected_radiation_density * dx)
expected_total = np.sum(expected_radiation)
np.testing.assert_array_equal(radiation_initial, 0.0)
np.testing.assert_array_equal(material_initial, 0.0)
radiation_atol = max(
    np.finfo(np.float64).tiny,
    residual_multiplier * cross_eps * expected_total,
)
np.testing.assert_allclose(
    radiation_final,
    expected_radiation,
    rtol=cross_analytic_rtol,
    atol=radiation_atol,
)
assert np.sum(radiation_final) > 0.0
q = np.sum(radiation_initial) - np.sum(radiation_final)
assert q < 0.0
np.testing.assert_allclose(q, -expected_total, rtol=ledger_rtol, atol=radiation_atol)

# d is measured independently from the EOS fields, not inferred from q.
initial_internal_energy = pressure_initial / (gamma - 1.0)
final_internal_energy = pressure_final / (gamma - 1.0)
eos_material_change = (final_internal_energy - initial_internal_energy) * dx
material_change = material_final - material_initial
assert np.all(material_change == 0.0)
eos_atol = (
    residual_multiplier
    * field_eps
    * max(
        np.max(np.abs(initial_internal_energy)),
        np.max(np.abs(final_internal_energy)),
    )
)
np.testing.assert_allclose(
    material_change, eos_material_change, rtol=0.0, atol=eos_atol
)
d = np.sum(material_change)
assert d == 0.0

assert radiation_table.shape[0] == 3
assert radiation_table.shape[1] >= 17
residual = radiation_table[-1, -2]
cumulative_residual = radiation_table[-1, -1]
table_scale = max(
    abs(radiation_table[0, 2]),
    abs(radiation_table[-1, 2]),
    abs(q),
    abs(d),
    abs(radiation_table[-1, 5]),
    abs(radiation_table[-1, 6]),
    abs(radiation_table[-1, 9]),
)
residual_atol = max(
    np.finfo(np.float64).tiny,
    residual_multiplier * cross_eps * table_scale,
)
np.testing.assert_allclose(residual, q - d, rtol=0.0, atol=residual_atol)
np.testing.assert_allclose(cumulative_residual, residual, rtol=0.0, atol=residual_atol)
assert abs(residual) > 0.5 * abs(q)
np.testing.assert_allclose(radiation_table[:2, -2:], 0.0, rtol=0.0, atol=residual_atol)
np.testing.assert_allclose(radiation_table[-1, 5], d, rtol=0.0, atol=residual_atol)
np.testing.assert_allclose(radiation_table[-1, 6], d, rtol=0.0, atol=residual_atol)
np.testing.assert_allclose(radiation_table[-1, 9], d, rtol=0.0, atol=residual_atol)
np.testing.assert_allclose(radiation_table[:, 7:9], 0.0, atol=residual_atol)
np.testing.assert_allclose(radiation_table[:, 10:-2], 0.0, atol=residual_atol)
np.testing.assert_allclose(
    radiation_table[0, 2] - radiation_table[-1, 2],
    residual,
    rtol=0.0,
    atol=residual_atol,
)

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
    )
    cross_compare_atol = max(
        np.finfo(np.float64).tiny,
        residual_multiplier * cross_eps * compare_scale,
    )
    for name in field_names:
        np.testing.assert_allclose(
            final[name], reference_final[name], rtol=0.0, atol=cross_compare_atol
        )
        np.testing.assert_allclose(
            initial[name], reference_initial[name], rtol=0.0, atol=cross_compare_atol
        )
    np.testing.assert_allclose(
        radiation_table, reference_table, rtol=0.0, atol=cross_compare_atol
    )

print(
    "1D hybrid near-floor realization: "
    f"Q={-q:.16e} J, q={q:.16e} J, d={d:.16e} J, "
    f"R={residual:.16e} J, cumulative_R={cumulative_residual:.16e} J"
)
