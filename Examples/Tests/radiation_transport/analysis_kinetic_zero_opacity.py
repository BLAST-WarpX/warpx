#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check the zero-opacity kinetic control has no requested or realized transfer."""

from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer


def load_fields(path: Path) -> dict[str, np.ndarray]:
    with (path / "Header").open() as header:
        header.readline()
        num_fields = int(header.readline())
        field_names = [header.readline().strip() for _ in range(num_fields)]
    fields = _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), field_names)
    return {name: np.asarray(value) for name, value in fields.items()}


initial_fields = load_fields(Path("diags/diag1000000"))
final_fields = load_fields(Path("diags/diag1000001"))
initial_material = initial_fields["radiation_material_energy"]
material = final_fields["radiation_material_energy"]
particle_energy = np.atleast_2d(np.loadtxt("diags/particle_energy.txt"))
radiation_energy = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))

assert radiation_energy.shape[1] >= 17
assert particle_energy.shape[1] >= 4

material_change = material - initial_material
np.testing.assert_array_equal(material_change, 0.0)
np.testing.assert_array_equal(material, 0.0)
np.testing.assert_array_equal(
    radiation_energy[:, 2:5],
    np.broadcast_to(radiation_energy[0, 2:5], radiation_energy[:, 2:5].shape),
)
np.testing.assert_array_equal(radiation_energy[:, 5:], 0.0)
np.testing.assert_array_equal(
    particle_energy[:, 2:],
    np.broadcast_to(particle_energy[0, 2:], particle_energy[:, 2:].shape),
)

realized = np.sum(material_change)
residual = radiation_energy[-1, 15]
electron_change = particle_energy[-1, 3] - particle_energy[0, 3]

print(
    "1D kinetic zero-opacity control: "
    f"realized={realized:.16e} J, residual={residual:.16e} J"
)
