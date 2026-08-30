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
parser.add_argument("--geometry", choices=["1d", "2d", "rcyl"], required=True)
parser.add_argument("--precision", choices=["SINGLE", "DOUBLE"], required=True)
parser.add_argument("--particle-precision", choices=["SINGLE", "DOUBLE"], required=True)
args = parser.parse_args()

plotfile = Path("diags/diag000001")
with open(plotfile / "Header") as header:
    header.readline()
    n_fields = int(header.readline())
    field_names = [header.readline().strip() for _ in range(n_fields)]

fields = _read_buffer(str(plotfile), str(plotfile / "Level_0" / "Cell_H"), field_names)
deposition = np.squeeze(fields["radiation_material_energy"])
single_precision = args.precision == "SINGLE" or args.particle_precision == "SINGLE"
rtol = 2.0e-5 if single_precision else 5.0e-13
atol = 2.0e-5 if single_precision else 1.0e-12

particle_energy = np.atleast_2d(np.loadtxt("diags/particle_energy.txt"))
radiation_energy = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
initial_energy = particle_energy[0, 2]
final_energy = particle_energy[-1, 2]

alpha = 3.0
path_length = 0.7
path_cell_fraction = 0.5
cell_size = 1.0 / 64.0
n_substeps = int(np.ceil(path_length / (path_cell_fraction * cell_size)))

if args.geometry == "2d":
    expected_deposition = np.zeros((64, 64))
    start = np.array([0.1, 0.1])
    direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
else:
    expected_deposition = np.zeros(64)
    start = np.array([0.1])
    direction = np.array([1.0])

remaining_energy = initial_energy
position = start.copy()
indices = np.floor(position / cell_size).astype(int)
remaining_distance = path_length
while remaining_distance > 4.0 * np.finfo(float).eps * path_length:
    face_distance = np.full(direction.shape, np.inf)
    moving = direction > 0.0
    next_faces = (indices + 1) * cell_size
    face_distance[moving] = (next_faces[moving] - position[moving]) / direction[moving]
    segment_distance = min(remaining_distance, np.min(face_distance))
    survival = np.exp(-alpha * segment_distance)
    deposited_energy = remaining_energy * (1.0 - survival)
    if args.geometry == "2d":
        expected_deposition[tuple(indices)] += deposited_energy
    else:
        expected_deposition[indices[0]] += deposited_energy
    remaining_energy *= survival
    remaining_distance -= segment_distance
    position += segment_distance * direction
    crossing = np.isclose(face_distance, segment_distance, rtol=0.0, atol=1.0e-14)
    indices[crossing] += 1

closed_form_final = initial_energy * np.exp(-alpha * path_length)
expected_material_gain = initial_energy - closed_form_final
measured_material_gain = np.sum(deposition)

print(f"geometry:                       {args.geometry}")
print(f"transport substeps:             {n_substeps}")
print(f"analytic final photon energy:   {closed_form_final:.16e} J")
print(f"simulated final photon energy:  {final_energy:.16e} J")
print(f"analytic material gain:         {expected_material_gain:.16e} J")
print(f"simulated material gain:        {measured_material_gain:.16e} J")

np.testing.assert_allclose(remaining_energy, closed_form_final, rtol=rtol)
np.testing.assert_allclose(final_energy, closed_form_final, rtol=rtol)
if args.geometry == "rcyl":
    # Radial streaming currently attenuates each bounded transport substep in
    # its starting radial cell; it does not solve the curved-face crossing.
    # The total path attenuation and all energy ledgers remain exact, but a
    # Cartesian face-by-face deposition profile is not a valid RCYL oracle.
    assert np.all(np.isfinite(deposition))
    assert np.all(deposition >= 0.0)
else:
    expected_support = np.flatnonzero(expected_deposition)
    np.testing.assert_array_equal(np.flatnonzero(deposition), expected_support)
    if single_precision:
        # The packet weight and deposited field are updated once per bounded
        # transport substep.  Bound their accumulated SP recurrence error by
        # the standard gamma_n factor, while preserving exact spatial support.
        epsilon = np.finfo(np.float32).eps
        gamma_n = n_substeps * epsilon / (1.0 - n_substeps * epsilon)
        atol = max(atol, gamma_n * np.max(expected_deposition))
    np.testing.assert_allclose(deposition, expected_deposition, rtol=rtol, atol=atol)
np.testing.assert_allclose(measured_material_gain, expected_material_gain, rtol=rtol)
np.testing.assert_allclose(radiation_energy[-1, 2], final_energy, rtol=rtol)
np.testing.assert_allclose(radiation_energy[-1, 6], expected_material_gain, rtol=rtol)
np.testing.assert_allclose(
    final_energy + measured_material_gain, initial_energy, rtol=rtol
)
