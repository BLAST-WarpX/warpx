#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer


def load_plotfile(path: Path) -> dict[str, np.ndarray]:
    with open(path / "Header") as header:
        header.readline()
        n_fields = int(header.readline())
        field_names = [header.readline().strip() for _ in range(n_fields)]
    return _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), field_names)


def load_table(path: Path) -> np.ndarray:
    table = np.loadtxt(path)
    return np.atleast_2d(table)


plotfile = Path("diags/diag1000001")
fields = load_plotfile(plotfile)
radiation = load_table(Path("diags/radiation_energy.txt"))
particle_energy = load_table(Path("diags/particle_energy.txt"))

single_precision = fields["radiation_material_energy"].dtype == np.float32
precision = np.float32 if single_precision else np.float64


def field_sum(field: np.ndarray) -> np.float64:
    return np.sum(np.asarray(field, dtype=np.float64), dtype=np.float64)


material_gain = field_sum(fields["radiation_material_energy"])
if "radiation_diffusion_energy" in fields:
    diffusion_group_energies = [field_sum(fields["radiation_diffusion_energy"])]
else:
    diffusion_field_names = sorted(
        name for name in fields if name.startswith("radiation_diffusion_energy_g")
    )
    diffusion_group_energies = [
        field_sum(fields[name]) for name in diffusion_field_names
    ]
diffusion_energy = np.sum(diffusion_group_energies, dtype=np.float64)
electron_loss = particle_energy[0, 3] - particle_energy[-1, 3]
photon_gain = particle_energy[-1, 4] - particle_energy[0, 4]
radiation_gain = diffusion_energy + photon_gain
requested_material = -radiation_gain
residual = radiation[-1, -2]
cumulative_residual = radiation[-1, -1]

assert radiation.shape[1] >= 17
assert radiation.shape[0] >= 2

field_roundoff = (
    32.0
    * np.finfo(precision).eps
    * max(
        abs(radiation[0, 2]),
        abs(radiation[-1, 2]),
        abs(radiation[-1, 4]),
        abs(radiation[-1, 5]),
        abs(radiation[-1, 6]),
        abs(radiation[-1, -8]),
        abs(material_gain),
        abs(requested_material),
    )
)
particle_roundoff = (
    32.0
    * np.finfo(precision).eps
    * max(
        np.max(np.abs(particle_energy[:, 3])),
        np.max(np.abs(radiation[:, 2])),
        abs(material_gain),
    )
)
field_atol = max(1.0e-24, field_roundoff)
particle_atol = max(1.0e-24, particle_roundoff)

print(f"RZ material-energy ledger: {material_gain:.16e} J")
print(f"RZ diffusion energy:       {diffusion_energy:.16e} J")
print(f"RZ streaming-photon gain:  {photon_gain:.16e} J")
print(f"RZ kinetic-electron loss:  {electron_loss:.16e} J")
print(f"RZ kinetic numerical residual: {residual:.16e} J")

assert diffusion_energy > 0.0
assert all(group_energy > 0.0 for group_energy in diffusion_group_energies)
assert photon_gain > 0.0
np.testing.assert_allclose(
    residual, requested_material - material_gain, rtol=0.0, atol=field_atol
)
np.testing.assert_allclose(cumulative_residual, residual, rtol=0.0, atol=field_atol)
np.testing.assert_allclose(radiation[-1, 5], material_gain, rtol=0.0, atol=field_atol)
np.testing.assert_allclose(radiation[-1, 6], material_gain, rtol=0.0, atol=field_atol)
np.testing.assert_allclose(radiation[-1, -8], material_gain, rtol=0.0, atol=field_atol)
np.testing.assert_allclose(electron_loss, -material_gain, rtol=0.0, atol=particle_atol)
np.testing.assert_allclose(
    electron_loss, radiation_gain + residual, rtol=0.0, atol=particle_atol
)
electron_radiation_drift = (
    particle_energy[-1, 3] - particle_energy[0, 3] + radiation[-1, 2] - radiation[0, 2]
)
np.testing.assert_allclose(
    electron_radiation_drift, -residual, rtol=0.0, atol=particle_atol
)
