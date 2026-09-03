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
parser.add_argument("--precision", choices=["SINGLE", "DOUBLE"], required=True)
parser.add_argument("--particle-precision", choices=["SINGLE", "DOUBLE"], required=True)
args = parser.parse_args()

plotfile = Path("diags/diag1000001")
with open(plotfile / "Header") as header:
    header.readline()
    n_fields = int(header.readline())
    field_names = [header.readline().strip() for _ in range(n_fields)]

fields = _read_buffer(str(plotfile), str(plotfile / "Level_0" / "Cell_H"), field_names)
single_precision = args.precision == "SINGLE" or args.particle_precision == "SINGLE"
rtol = 5.0e-6 if single_precision else 3.0e-13
particle_energy = np.loadtxt("diags/particle_energy.txt")
radiation_energy = np.loadtxt("diags/radiation_energy.txt")

initial_photon_energy = particle_energy[0, 4]
final_photon_energy = particle_energy[-1, 4]
diffusion_energy = np.sum(fields["radiation_diffusion_energy"])
material_energy = np.sum(fields["radiation_material_energy"])

print(f"initial streaming energy: {initial_photon_energy:.16e} J")
print(f"final streaming energy:   {final_photon_energy:.16e} J")
print(f"final diffusion energy:   {diffusion_energy:.16e} J")

assert diffusion_energy > 0.0
assert final_photon_energy < initial_photon_energy
np.testing.assert_allclose(material_energy, 0.0, atol=1.0e-15)
np.testing.assert_allclose(
    final_photon_energy + diffusion_energy,
    initial_photon_energy,
    rtol=rtol,
)
np.testing.assert_allclose(radiation_energy[-1, 2], initial_photon_energy, rtol=rtol)
np.testing.assert_allclose(radiation_energy[-1, 3], final_photon_energy, rtol=rtol)
np.testing.assert_allclose(radiation_energy[-1, 4], diffusion_energy, rtol=rtol)
np.testing.assert_allclose(radiation_energy[-1, 5], 0.0, atol=1.0e-15)
np.testing.assert_allclose(radiation_energy[-1, 6], 0.0, atol=1.0e-15)
np.testing.assert_allclose(radiation_energy[-1, 7:17], 0.0, atol=1.0e-15)
