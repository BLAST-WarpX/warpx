#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Analytic one-step FLD transport/recoil with an absorption control."""

import argparse
from pathlib import Path

import numpy as np
import yt
from read_raw_data import _read_buffer
from scipy.constants import c, m_p

RADIATION_CONSTANT = 7.565733250280007e-16
NUM_CELLS = 16
DX = 1.0 / NUM_CELLS
DT = 1.0e-12
ALPHA_TRANSPORT = 64.0


def load_fields(path):
    with open(path / "Header") as header:
        header.readline()
        num_fields = int(header.readline())
        field_names = [header.readline().strip() for _ in range(num_fields)]
    fields = _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), field_names)
    return {
        name: np.asarray(values, dtype=np.float64).squeeze()
        for name, values in fields.items()
    }


def fld_step(initial_energy):
    """Reproduce one conservative finite-volume FLD substep."""
    density = initial_energy / DX
    rate = np.zeros(NUM_CELLS)
    cell_flux = np.zeros(NUM_CELLS)
    for cell in range(NUM_CELLS):
        for side in (-1, 1):
            neighbor = cell + side
            if neighbor < 0 or neighbor >= NUM_CELLS:
                continue  # reflecting boundary
            gradient = (density[neighbor] - density[cell]) / DX
            face_density = max(0.0, 0.5 * (density[cell] + density[neighbor]))
            if face_density == 0.0:
                continue
            dimensionless_gradient = abs(gradient) / (ALPHA_TRANSPORT * face_density)
            limiter = (2.0 + dimensionless_gradient) / (
                6.0 + 3.0 * dimensionless_gradient + dimensionless_gradient**2
            )
            diffusion_coefficient = c * limiter / ALPHA_TRANSPORT
            rate[cell] += diffusion_coefficient * gradient
            cell_flux[cell] -= 0.5 * side * diffusion_coefficient * gradient
    transported = initial_energy + DT * rate
    impulse = DT * DX * ALPHA_TRANSPORT * cell_flux / c
    return transported, impulse


def particle_moments(path):
    dataset = yt.load(str(path))
    particles = dataset.all_data()
    # yt maps the sole mesh coordinate of a 1D-Z WarpX plotfile to x.
    positions = np.asarray(
        particles["ions", "particle_position_x"].to_value("m"), dtype=np.float64
    )
    weights = np.asarray(particles["ions", "particle_weight"].v, dtype=np.float64)
    momenta = np.stack(
        [
            np.asarray(
                particles["ions", f"particle_momentum_{axis}"].to_value("kg*m/s"),
                dtype=np.float64,
            )
            for axis in ("x", "y", "z")
        ],
        axis=1,
    )
    cells = np.minimum((positions / DX).astype(int), NUM_CELLS - 1)
    mass = np.bincount(cells, weights=weights * m_p, minlength=NUM_CELLS)
    momentum = np.stack(
        [
            np.bincount(
                cells, weights=weights * momenta[:, component], minlength=NUM_CELLS
            )
            for component in range(3)
        ],
        axis=1,
    )
    momentum_squared = np.sum(momenta * momenta, axis=1)
    kinetic_per_particle = (
        momentum_squared
        / m_p
        / (1.0 + np.sqrt(1.0 + momentum_squared / (m_p * c) ** 2))
    )
    kinetic = np.bincount(
        cells, weights=weights * kinetic_per_particle, minlength=NUM_CELLS
    )
    return mass, momentum, kinetic


parser = argparse.ArgumentParser()
parser.add_argument("--planck-absorption", type=float, required=True)
parser.add_argument("--precision", choices=["SINGLE", "DOUBLE"], default="DOUBLE")
parser.add_argument("--reference", type=Path)
args = parser.parse_args()

initial = load_fields(Path("diags/diag000001"))
final = load_fields(Path("diags/diag000002"))
radiation = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
momentum_ledger = np.atleast_2d(np.loadtxt("diags/radiation_momentum.txt"))

initial_radiation = initial["radiation_diffusion_energy"]
temperature = initial["Te"]
assert initial_radiation.shape == (NUM_CELLS,)
assert np.all(initial_radiation[:4] > 0.0)
np.testing.assert_array_equal(initial_radiation[4:], 0.0)

planck_fraction = -np.expm1(-c * args.planck_absorption * DT)
equilibrium = RADIATION_CONSTANT * temperature**4 * DX
after_lte = initial_radiation.copy()
after_lte[1] += planck_fraction * (equilibrium[1] - initial_radiation[1])
expected_internal_exchange = initial_radiation - after_lte
# The LTE request is cell-local, while the hybrid EOS update maps through
# nodes and records the authoritative old-to-final cell realization.
expected_realized_internal_exchange = np.zeros_like(expected_internal_exchange)
expected_realized_internal_exchange[:3] = expected_internal_exchange[1] * np.array(
    [0.25, 0.5, 0.25]
)
np.testing.assert_allclose(
    np.sum(expected_realized_internal_exchange),
    np.sum(expected_internal_exchange),
    rtol=0.0,
    atol=np.finfo(np.float64).eps * np.max(initial_radiation),
)

after_diffusion, expected_impulse = fld_step(after_lte)
mass, measured_cell_momentum, measured_cell_kinetic = particle_moments(
    Path("diags/diag000002")
)
proper_velocity = expected_impulse / mass
expected_kinetic = (
    mass * proper_velocity**2 / (1.0 + np.sqrt(1.0 + (proper_velocity / c) ** 2))
)
expected_final_radiation = after_diffusion - expected_kinetic

if args.precision == "SINGLE":
    field_rtol = 8.0e-5
    ledger_rtol = 1.0e-4
    impulse_rtol = 2.0e-5
else:
    field_rtol = 3.0e-11
    ledger_rtol = 2.0e-11
    impulse_rtol = 2.0e-10

np.testing.assert_allclose(
    final["radiation_diffusion_energy"],
    expected_final_radiation,
    rtol=field_rtol,
    atol=field_rtol * np.max(initial_radiation),
)
np.testing.assert_allclose(
    final["radiation_material_energy"],
    expected_realized_internal_exchange,
    rtol=ledger_rtol,
    atol=ledger_rtol * np.max(initial_radiation),
)
np.testing.assert_allclose(
    measured_cell_momentum[:, 2],
    expected_impulse,
    rtol=impulse_rtol,
    atol=impulse_rtol * np.max(np.abs(expected_impulse)),
)
np.testing.assert_allclose(
    measured_cell_momentum[:, :2],
    0.0,
    atol=impulse_rtol * np.max(np.abs(expected_impulse)),
)
np.testing.assert_allclose(
    measured_cell_kinetic,
    expected_kinetic,
    rtol=field_rtol,
    atol=field_rtol * np.max(expected_kinetic),
)

expected_internal = np.sum(expected_realized_internal_exchange)
expected_work = np.sum(expected_kinetic)
expected_momentum = np.sum(expected_impulse)
stage_initial = radiation[-2, 2]
np.testing.assert_allclose(stage_initial, np.sum(initial_radiation), rtol=ledger_rtol)
np.testing.assert_allclose(radiation[-1, 3], 0.0, atol=1.0e-30)
np.testing.assert_allclose(
    radiation[-1, 4], np.sum(expected_final_radiation), rtol=ledger_rtol
)
np.testing.assert_allclose(radiation[-1, 9], expected_internal, rtol=ledger_rtol)
np.testing.assert_allclose(radiation[-1, 10], expected_work, rtol=ledger_rtol)
np.testing.assert_allclose(
    radiation[-1, 5], expected_internal + expected_work, rtol=ledger_rtol
)
np.testing.assert_allclose(radiation[-2, 6], -stage_initial, rtol=ledger_rtol)
np.testing.assert_allclose(
    radiation[-1, 6],
    -stage_initial + expected_internal + expected_work,
    rtol=ledger_rtol,
)
np.testing.assert_allclose(
    radiation[-1, 2] + radiation[-1, 5], stage_initial, rtol=ledger_rtol
)
np.testing.assert_allclose(radiation[-1, 2] + radiation[-1, 6], 0.0, atol=1.0e-11)
np.testing.assert_allclose(momentum_ledger[-1, 4], expected_momentum, rtol=impulse_rtol)
np.testing.assert_allclose(momentum_ledger[-1, 7], expected_momentum, rtol=impulse_rtol)
np.testing.assert_allclose(momentum_ledger[-1, [2, 3, 5, 6]], 0.0, atol=1.0e-24)
np.testing.assert_allclose(momentum_ledger[-1, 8:20], 0.0, atol=1.0e-24)

assert np.count_nonzero(final["radiation_diffusion_energy"] > 0.0) > 1
assert abs(expected_momentum) > 0.0
assert expected_work > 0.0
if args.planck_absorption == 0.0:
    np.testing.assert_array_equal(expected_internal_exchange, 0.0)
    np.testing.assert_array_equal(final["radiation_material_energy"], 0.0)
    np.testing.assert_allclose(radiation[-1, 9], 0.0, atol=1.0e-30)
else:
    assert expected_internal > 0.0
    assert radiation[-1, 9] > 0.0
    assert np.flatnonzero(final["radiation_material_energy"]).tolist() == [0, 1, 2]
    assert np.all(final["radiation_material_energy"][:3] > 0.0)

if args.reference is not None:
    reference_initial = load_fields(args.reference / "diags/diag000001")
    reference_final = load_fields(args.reference / "diags/diag000002")
    reference_radiation = np.atleast_2d(
        np.loadtxt(args.reference / "diags/radiation_energy.txt")
    )
    np.testing.assert_array_equal(
        initial_radiation, reference_initial["radiation_diffusion_energy"]
    )
    np.testing.assert_array_equal(reference_final["radiation_material_energy"], 0.0)
    assert radiation[-1, 9] > reference_radiation[-1, 9]
    assert radiation[-1, 2] < reference_radiation[-1, 2]

print(f"stage-initial diffusion energy: {stage_initial:.16e} J")
print(f"Planck material exchange:       {expected_internal:.16e} J")
print(f"Rosseland material impulse:     {expected_momentum:.16e} kg*m/s")
print(f"recoil kinetic work:            {expected_work:.16e} J")
print(f"final diffusion energy:         {radiation[-1, 4]:.16e} J")
