#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse
from pathlib import Path

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from read_raw_data import _read_buffer
from scipy.constants import c

parser = argparse.ArgumentParser()
add_precision_arguments(parser)
args = parser.parse_args()
_, _, cross_dtype = precision_dtypes(args)

plotfile = Path("diags/diag1000001")
with open(plotfile / "Header") as header:
    header.readline()
    n_fields = int(header.readline())
    field_names = [header.readline().strip() for _ in range(n_fields)]

fields = _read_buffer(str(plotfile), str(plotfile / "Level_0" / "Cell_H"), field_names)
cross_rtol = 5.0e-6 if cross_dtype == np.float32 else 3.0e-13
particle_energy = np.loadtxt("diags/particle_energy.txt")

alpha_lo = 5.0
alpha_hi = 20.0
dt = 4.0e-10
cell_size = 1.0 / 16.0
path_cell_fraction = 0.5
initial_z = 0.42

n_substeps = int(np.ceil(c * dt / (path_cell_fraction * cell_size)))
initial_photon_energy = particle_energy[0, 3]
expected_deposition = np.zeros(16)
remaining_energies = np.array(
    [initial_photon_energy / 3.0, 2.0 * initial_photon_energy / 3.0]
)
for energy_group, opacity_multiplier in enumerate((1.0, 2.0)):
    z = initial_z
    cell = int(np.floor(z / cell_size))
    remaining_distance = c * dt
    while remaining_distance > 4.0 * np.finfo(float).eps * c * dt:
        distance_to_face = (cell + 1) * cell_size - z
        segment_distance = min(remaining_distance, distance_to_face)
        alpha = opacity_multiplier * (alpha_lo if cell < 8 else alpha_hi)
        survival = np.exp(-alpha * segment_distance)
        deposited_energy = remaining_energies[energy_group] * (1.0 - survival)
        expected_deposition[cell] += deposited_energy
        remaining_energies[energy_group] *= survival
        remaining_distance -= segment_distance
        z += segment_distance
        if np.isclose(segment_distance, distance_to_face, rtol=0.0, atol=1.0e-14):
            cell += 1

remaining_energy = np.sum(remaining_energies)

deposition = fields["radiation_material_energy"].squeeze()

print(f"transport substeps:            {n_substeps}")
print(f"traversed material cells:      {np.count_nonzero(deposition)}")
print(f"analytic final photon energy:  {remaining_energy:.16e} J")
print(f"simulated final photon energy: {particle_energy[-1, 3]:.16e} J")

np.testing.assert_allclose(
    deposition, expected_deposition, rtol=cross_rtol, atol=1.0e-13
)
np.testing.assert_allclose(particle_energy[-1, 3], remaining_energy, rtol=cross_rtol)
assert np.count_nonzero(deposition) == 3
