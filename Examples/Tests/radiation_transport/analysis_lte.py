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
from scipy.constants import Boltzmann, c, elementary_charge, physical_constants

parser = argparse.ArgumentParser()
parser.add_argument("--coupling", choices=["hybrid", "kinetic"], required=True)
parser.add_argument("--diffusion", action="store_true")
parser.add_argument("--conversion", action="store_true")
add_precision_arguments(parser)
args = parser.parse_args()
_, _, cross_dtype = precision_dtypes(args)


def load_plotfile(path: Path) -> dict[str, np.ndarray]:
    with open(path / "Header") as header:
        header.readline()
        n_fields = int(header.readline())
        field_names = [header.readline().strip() for _ in range(n_fields)]
    return _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), field_names)


initial_fields = load_plotfile(Path("diags/diag1000000"))
fields = load_plotfile(Path("diags/diag1000001"))
cross_analytic_rtol = 5.0e-6 if cross_dtype == np.float32 else 3.0e-13
hybrid_conservation_rtol = 5.0e-6 if cross_dtype == np.float32 else 2.0e-12
kinetic_conservation_rtol = 5.0e-6 if cross_dtype == np.float32 else 2.0e-11
material_gain = np.sum(fields["radiation_material_energy"])
realized_material = np.sum(
    fields["radiation_material_energy"] - initial_fields["radiation_material_energy"]
)
diffusion_energy_gain = np.sum(fields["radiation_diffusion_energy"])
photon_energy_gain = 0.0
if args.conversion:
    particle_energy = np.loadtxt("diags/particle_energy.txt")
    photon_energy_gain = particle_energy[-1, 4] - particle_energy[0, 4]
radiation_gain = diffusion_energy_gain + photon_energy_gain

print(f"material-energy ledger: {material_gain:.16e} J")
print(f"diffusion-energy gain:   {diffusion_energy_gain:.16e} J")
print(f"streaming-photon gain:   {photon_energy_gain:.16e} J")

assert radiation_gain > 0.0
if args.coupling == "hybrid":
    np.testing.assert_allclose(material_gain, -radiation_gain, rtol=cross_analytic_rtol)
else:
    radiation_table = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
    assert radiation_table.shape[1] >= 17
    requested_material = -radiation_gain
    residual = radiation_table[-1, -2]
    cumulative_residual = radiation_table[-1, -1]
    closure_residual = (
        radiation_table[0, 2] - radiation_table[-1, 2] - realized_material
    )
    precision = cross_dtype
    field_roundoff = (
        32.0
        * np.finfo(precision).eps
        * max(
            abs(radiation_table[0, 2]),
            abs(radiation_table[-1, 2]),
            abs(radiation_table[-1, 5]),
            abs(radiation_table[-1, 6]),
            abs(realized_material),
        )
    )
    np.testing.assert_allclose(
        residual, closure_residual, rtol=0.0, atol=max(1.0e-24, field_roundoff)
    )
    np.testing.assert_allclose(
        residual,
        requested_material - realized_material,
        rtol=0.0,
        atol=max(1.0e-24, field_roundoff),
    )
    np.testing.assert_allclose(
        cumulative_residual, residual, rtol=0.0, atol=max(1.0e-24, field_roundoff)
    )
    np.testing.assert_allclose(
        radiation_table[-1, 5],
        realized_material,
        rtol=0.0,
        atol=max(1.0e-24, field_roundoff),
    )
    np.testing.assert_allclose(
        radiation_table[-1, 6],
        realized_material,
        rtol=0.0,
        atol=max(1.0e-24, field_roundoff),
    )

if args.coupling == "hybrid":
    alpha_planck = 20.0
    dt = 1.0e-10
    initial_temperature = elementary_charge / Boltzmann
    gamma = 5.0 / 3.0
    radiation_constant = 4.0 * physical_constants["Stefan-Boltzmann constant"][0] / c
    expected_radiation_gain = (
        radiation_constant
        * initial_temperature**4
        * (1.0 - np.exp(-alpha_planck * c * dt))
    )
    if args.diffusion:
        expected_radiation_gain /= 16.0
    electron_density = fields["rho"] / elementary_charge
    electron_energy_loss = np.sum(
        electron_density
        * Boltzmann
        * (initial_temperature - fields["Te"])
        / (gamma - 1.0)
        / 16.0
    )
    print(f"analytic radiation gain: {expected_radiation_gain:.16e} J")
    print(f"hybrid-electron loss:    {electron_energy_loss:.16e} J")
    np.testing.assert_allclose(
        radiation_gain, expected_radiation_gain, rtol=cross_analytic_rtol
    )
    np.testing.assert_allclose(
        electron_energy_loss, radiation_gain, rtol=hybrid_conservation_rtol
    )
    if args.diffusion:
        diffusion_energy = fields["radiation_diffusion_energy"].squeeze()
        assert np.all(diffusion_energy >= 0.0)
        assert np.count_nonzero(diffusion_energy) > 1
    if args.conversion:
        assert photon_energy_gain > 0.0
else:
    particle_energy = np.loadtxt("diags/particle_energy.txt")
    electron_energy_loss = particle_energy[0, 3] - particle_energy[-1, 3]
    print(f"kinetic-electron loss:   {electron_energy_loss:.16e} J")
    np.testing.assert_allclose(
        electron_energy_loss,
        -realized_material,
        rtol=kinetic_conservation_rtol,
        atol=max(1.0e-24, field_roundoff),
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
    print(f"kinetic numerical residual: {residual:.16e} J")
    assert particle_energy[-1, 3] >= 0.0
