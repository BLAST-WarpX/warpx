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
from scipy.constants import Boltzmann, c, elementary_charge

parser = argparse.ArgumentParser()
parser.add_argument("plotfile")
parser.add_argument("--coupling", choices=["hybrid", "kinetic"], required=True)
parser.add_argument("--require-multibox-residual", action="store_true")
add_precision_arguments(parser)
args = parser.parse_args()
field_dtype, _, cross_dtype = precision_dtypes(args)

plotfile = Path(args.plotfile)

with open(plotfile / "Header") as header:
    header.readline()
    n_fields = int(header.readline())
    field_names = [header.readline().strip() for _ in range(n_fields)]

fields = _read_buffer(str(plotfile), str(plotfile / "Level_0" / "Cell_H"), field_names)
cross_analytic_rtol = 2.0e-6 if cross_dtype == np.float32 else 2.0e-13
conservation_rtol = 5.0e-6 if cross_dtype == np.float32 else 2.0e-12
particle_energy = np.loadtxt("diags/particle_energy.txt")

alpha = 10.0
dt = 1.0e-10
gamma = 5.0 / 3.0
cell_size = 1.0 / 16.0
initial_temperature = 100.0 * elementary_charge / Boltzmann

initial_photon_energy = particle_energy[0, 4]
final_photon_energy = particle_energy[-1, 4]
expected_final_photon_energy = initial_photon_energy * np.exp(-alpha * c * dt)
photon_energy_loss = initial_photon_energy - final_photon_energy

absorbed_energy = np.sum(fields["radiation_material_energy"])
if args.coupling == "hybrid":
    electron_density = fields["rho"] / elementary_charge
    electron_energy_gain = np.sum(
        electron_density
        * Boltzmann
        * (fields["Te"] - initial_temperature)
        / (gamma - 1.0)
        * cell_size
    )
else:
    electron_energy_gain = particle_energy[-1, 3] - particle_energy[0, 3]

print(f"analytic final photon energy: {expected_final_photon_energy:.16e} J")
print(f"simulated final photon energy: {final_photon_energy:.16e} J")
print(f"photon energy loss:            {photon_energy_loss:.16e} J")
print(f"absorbed-energy ledger:        {absorbed_energy:.16e} J")
print(f"electron energy gain:          {electron_energy_gain:.16e} J")

np.testing.assert_allclose(
    final_photon_energy, expected_final_photon_energy, rtol=cross_analytic_rtol
)
if args.coupling == "hybrid":
    np.testing.assert_allclose(
        absorbed_energy, photon_energy_loss, rtol=cross_analytic_rtol
    )
else:
    radiation_table = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
    assert radiation_table.shape[1] >= 17
    residual = radiation_table[-1, 15]
    closure_residual = radiation_table[0, 2] - radiation_table[-1, 2] - absorbed_energy
    precision = cross_dtype
    field_roundoff = (
        32.0
        * np.finfo(precision).eps
        * max(
            abs(radiation_table[0, 2]),
            abs(radiation_table[-1, 2]),
            abs(absorbed_energy),
        )
    )
    np.testing.assert_allclose(
        residual, closure_residual, rtol=0.0, atol=max(1.0e-24, field_roundoff)
    )
    electron_radiation_drift = (
        particle_energy[-1, 3]
        - particle_energy[0, 3]
        + radiation_table[-1, 2]
        - radiation_table[0, 2]
    )
    particle_roundoff = (
        32.0
        * np.finfo(precision).eps
        * max(
            abs(particle_energy[0, 3]),
            abs(particle_energy[-1, 3]),
            abs(radiation_table[0, 2]),
            abs(radiation_table[-1, 2]),
        )
    )
    np.testing.assert_allclose(
        electron_radiation_drift,
        -residual,
        rtol=0.0,
        atol=max(1.0e-24, particle_roundoff),
    )
    if args.require_multibox_residual:
        assert len(list((plotfile / "Level_0").glob("Cell_D_*"))) >= 2
        assert np.isfinite(residual)
        if field_dtype != np.float32:
            assert residual != 0.0
    print(f"kinetic numerical residual: {residual:.16e} J")
np.testing.assert_allclose(
    electron_energy_gain, absorbed_energy, rtol=conservation_rtol
)
