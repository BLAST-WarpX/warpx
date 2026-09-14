#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Analytic two-slab attenuation for registry-selected pure HDF5 tables."""

import argparse
from pathlib import Path

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from read_raw_data import _read_buffer

parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path)
parser.add_argument("--kinetic", action="store_true")
add_precision_arguments(parser)
args = parser.parse_args()
_, _, cross_dtype = precision_dtypes(args)

plotfile = Path("diags/diag000001")
with open(plotfile / "Header") as header:
    header.readline()
    n_fields = int(header.readline())
    field_names = [header.readline().strip() for _ in range(n_fields)]
fields = _read_buffer(str(plotfile), str(plotfile / "Level_0" / "Cell_H"), field_names)
deposition = np.squeeze(fields["radiation_material_energy"])
radiation = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))

if args.kinetic:
    initial_plotfile = Path("diags/diag000000")
    with open(initial_plotfile / "Header") as header:
        header.readline()
        n_initial_fields = int(header.readline())
        initial_field_names = [
            header.readline().strip() for _ in range(n_initial_fields)
        ]
    initial_fields = _read_buffer(
        str(initial_plotfile),
        str(initial_plotfile / "Level_0" / "Cell_H"),
        initial_field_names,
    )
    initial_deposition = np.squeeze(initial_fields["radiation_material_energy"])
    particle_energy = np.atleast_2d(np.loadtxt("diags/particle_energy.txt"))

initial_energy = radiation[0, 2]
alpha_a = 10.0 * 3.162277660168379e-2
alpha_b = 10.0 * 6.324555320336758e-2
cell_length = 0.025
expected = np.zeros(40)
remaining = initial_energy
for cell, alpha in [
    *((cell, alpha_a) for cell in range(8, 16)),
    *((cell, alpha_b) for cell in range(24, 32)),
]:
    after_cell = remaining * np.exp(-alpha * cell_length)
    expected[cell] = remaining - after_cell
    remaining = after_cell

cross_rtol = 8.0e-5 if cross_dtype == np.float32 else 1.0e-12
atol = 3.0e-12 * initial_energy
print(f"initial photon energy:          {initial_energy:.16e} J")
print(f"material-A deposition:          {np.sum(deposition[8:16]):.16e} J")
print(f"material-B deposition:          {np.sum(deposition[24:32]):.16e} J")
print(f"analytic final photon energy:   {remaining:.16e} J")
print(f"simulated final photon energy:  {radiation[-1, 2]:.16e} J")

np.testing.assert_allclose(radiation[-1, 2], remaining, rtol=cross_rtol)
active_cells = [*range(8, 16), *range(24, 32)]
if not args.kinetic:
    realized_total = np.sum(deposition)
    requested_transfer = initial_energy - radiation[-1, 2]
    residual = radiation[-1, -2]
    cumulative_residual = radiation[-1, -1]
    field_roundoff = (
        32.0
        * np.finfo(cross_dtype).eps
        * max(
            abs(initial_energy),
            abs(radiation[-1, 2]),
            abs(realized_total),
            abs(radiation[-1, 5]),
            abs(radiation[-1, 6]),
        )
    )
    tolerance = max(1.0e-24, field_roundoff)
    realized_cells = [*range(7, 17), *range(23, 33)]
    assert np.flatnonzero(deposition).tolist() == realized_cells
    assert np.all(deposition[realized_cells] > 0.0)
    np.testing.assert_allclose(requested_transfer, np.sum(expected), rtol=cross_rtol)
    np.testing.assert_allclose(
        radiation[-1, 5], realized_total, rtol=0.0, atol=tolerance
    )
    np.testing.assert_allclose(
        radiation[-1, 6], realized_total, rtol=0.0, atol=tolerance
    )
    material_column = radiation.shape[1] - 8
    np.testing.assert_allclose(
        radiation[-1, material_column], realized_total, rtol=0.0, atol=tolerance
    )
    np.testing.assert_allclose(
        residual,
        requested_transfer - realized_total,
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(cumulative_residual, residual, rtol=0.0, atol=tolerance)
    np.testing.assert_allclose(
        initial_energy - radiation[-1, 2] - realized_total,
        residual,
        rtol=0.0,
        atol=tolerance,
    )
    print(f"realized material transfer:    {realized_total:.16e} J")
    print(f"requested material transfer:   {requested_transfer:.16e} J")
    print(f"hybrid numerical residual:      {residual:.16e} J")
else:
    realized_material = deposition - initial_deposition
    realized_total = np.sum(realized_material)
    requested_transfer = initial_energy - radiation[-1, 2]
    residual = radiation[-1, -2]
    cumulative_residual = radiation[-1, -1]
    field_roundoff = (
        32.0
        * np.finfo(cross_dtype).eps
        * max(
            abs(initial_energy),
            abs(radiation[-1, 2]),
            abs(realized_total),
            abs(radiation[-1, 5]),
            abs(radiation[-1, 6]),
        )
    )
    particle_roundoff = (
        32.0
        * np.finfo(cross_dtype).eps
        * max(
            abs(particle_energy[0, 3]),
            abs(particle_energy[-1, 3]),
            abs(initial_energy),
            abs(radiation[-1, 2]),
            abs(realized_total),
        )
    )
    tolerance = max(1.0e-24, field_roundoff)
    particle_tolerance = max(1.0e-24, particle_roundoff)
    assert np.flatnonzero(realized_material).tolist() == active_cells
    assert np.all(realized_material[active_cells] > 0.0)
    np.testing.assert_allclose(requested_transfer, np.sum(expected), rtol=cross_rtol)
    np.testing.assert_allclose(
        radiation[-1, 5], realized_total, rtol=0.0, atol=tolerance
    )
    np.testing.assert_allclose(
        radiation[-1, 6], realized_total, rtol=0.0, atol=tolerance
    )
    material_column = radiation.shape[1] - 8
    np.testing.assert_allclose(
        radiation[-1, material_column], realized_total, rtol=0.0, atol=tolerance
    )
    np.testing.assert_allclose(
        residual,
        requested_transfer - realized_total,
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(cumulative_residual, residual, rtol=0.0, atol=tolerance)
    np.testing.assert_allclose(
        initial_energy - radiation[-1, 2] - realized_total,
        residual,
        rtol=0.0,
        atol=tolerance,
    )
    electron_gain = particle_energy[-1, 3] - particle_energy[0, 3]
    np.testing.assert_allclose(
        electron_gain, realized_total, rtol=0.0, atol=particle_tolerance
    )
    electron_radiation_drift = electron_gain + radiation[-1, 2] - initial_energy
    np.testing.assert_allclose(
        electron_radiation_drift,
        -residual,
        rtol=0.0,
        atol=particle_tolerance,
    )
    print(f"realized material transfer:    {realized_total:.16e} J")
    print(f"requested material transfer:   {requested_transfer:.16e} J")
    print(f"kinetic numerical residual:     {residual:.16e} J")

if args.reference is not None:
    reference_plotfile = args.reference / "diags/diag000001"
    with open(reference_plotfile / "Header") as header:
        header.readline()
        n_reference_fields = int(header.readline())
        reference_names = [header.readline().strip() for _ in range(n_reference_fields)]
    reference_fields = _read_buffer(
        str(reference_plotfile),
        str(reference_plotfile / "Level_0" / "Cell_H"),
        reference_names,
    )
    reference_deposition = np.squeeze(reference_fields["radiation_material_energy"])
    reference_radiation = np.atleast_2d(
        np.loadtxt(args.reference / "diags/radiation_energy.txt")
    )
    np.testing.assert_array_equal(deposition, reference_deposition)
    np.testing.assert_array_equal(radiation, reference_radiation)
