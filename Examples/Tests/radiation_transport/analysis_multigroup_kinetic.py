#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from read_raw_data import _read_buffer

parser = argparse.ArgumentParser()
add_precision_arguments(parser)
args = parser.parse_args()
_, _, cross_dtype = precision_dtypes(args)

radiation = np.loadtxt("diags/radiation_energy.txt")
particle = np.loadtxt("diags/particle_energy.txt")
final = radiation[-1]

plotfile = "diags/diag1000001"
initial_plotfile = "diags/diag1000000"


def load_fields(path: str) -> dict[str, np.ndarray]:
    with open(f"{path}/Header") as header:
        header.readline()
        n_fields = int(header.readline())
        field_names = [header.readline().strip() for _ in range(n_fields)]
    return _read_buffer(path, f"{path}/Level_0/Cell_H", field_names)


initial_fields = load_fields(initial_plotfile)
fields = load_fields(plotfile)
realized_material = np.sum(
    fields["radiation_material_energy"] - initial_fields["radiation_material_energy"]
)

diffusion_group_energy = final[9:11]
diffusion_energy = final[4]
material_exchange = final[5]
electron_energy_loss = particle[0, 3] - particle[-1, 3]

conservation_rtol = 8.0e-6 if cross_dtype == np.float32 else 2.0e-11
sum_rtol = 2.0e-6 if cross_dtype == np.float32 else 2.0e-15
grid_group_energy = np.array(
    [
        np.sum(fields["radiation_diffusion_energy_g0"]),
        np.sum(fields["radiation_diffusion_energy_g1"]),
    ]
)

print(f"diffusion group energies: {diffusion_group_energy}")
print(f"material exchange:        {material_exchange:.16e} J")
print(f"kinetic-electron loss:    {electron_energy_loss:.16e} J")

assert np.all(diffusion_group_energy > 0.0)
np.testing.assert_allclose(
    diffusion_energy, np.sum(diffusion_group_energy), rtol=sum_rtol
)
np.testing.assert_allclose(grid_group_energy, diffusion_group_energy, rtol=sum_rtol)
precision = cross_dtype
field_roundoff = (
    32.0
    * np.finfo(precision).eps
    * max(
        abs(radiation[0, 2]),
        abs(radiation[-1, 2]),
        abs(realized_material),
        abs(material_exchange),
        abs(radiation[-1, 6]),
    )
)
np.testing.assert_allclose(
    material_exchange,
    realized_material,
    rtol=0.0,
    atol=max(1.0e-24, field_roundoff),
)
np.testing.assert_allclose(
    radiation[-1, 6], realized_material, rtol=0.0, atol=max(1.0e-24, field_roundoff)
)
residual = radiation[-1, -2]
cumulative_residual = radiation[-1, -1]
closure_residual = radiation[0, 2] - radiation[-1, 2] - realized_material
requested_material = -diffusion_energy
np.testing.assert_allclose(
    residual,
    closure_residual,
    rtol=0.0,
    atol=max(1.0e-24, field_roundoff),
)
np.testing.assert_allclose(
    residual,
    requested_material - realized_material,
    rtol=0.0,
    atol=max(1.0e-24, field_roundoff),
)
np.testing.assert_allclose(
    cumulative_residual,
    residual,
    rtol=0.0,
    atol=max(1.0e-24, field_roundoff),
)
electron_radiation_drift = (
    particle[-1, 3] - particle[0, 3] + radiation[-1, 2] - radiation[0, 2]
)
particle_roundoff = (
    32.0
    * np.finfo(precision).eps
    * max(
        abs(particle[0, 3]),
        abs(particle[-1, 3]),
        abs(radiation[0, 2]),
        abs(radiation[-1, 2]),
    )
)
np.testing.assert_allclose(
    electron_energy_loss,
    -realized_material,
    rtol=conservation_rtol,
    atol=max(1.0e-24, particle_roundoff),
)
np.testing.assert_allclose(
    electron_radiation_drift,
    -residual,
    rtol=0.0,
    atol=max(1.0e-24, particle_roundoff),
)
print(f"kinetic numerical residual: {residual:.16e} J")
assert particle[-1, 3] >= 0.0
