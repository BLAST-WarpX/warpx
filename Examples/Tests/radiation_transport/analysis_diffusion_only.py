#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Validate conservative FLD without LTE or a material-state adapter."""

from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer


def load_plotfile(path: Path) -> dict[str, np.ndarray]:
    with open(path / "Header") as header:
        header.readline()
        n_fields = int(header.readline())
        field_names = [header.readline().strip() for _ in range(n_fields)]
    return _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), field_names)


fields = load_plotfile(Path("diags/diag1000001"))
radiation = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
diffusion = np.asarray(fields["radiation_diffusion_energy"]).squeeze()
material = np.asarray(fields["radiation_material_energy"]).squeeze()

assert diffusion.shape == (8,)
assert radiation.shape == (2, 17)
assert np.all(np.isfinite(diffusion))
assert np.all(diffusion >= 0.0)

# The cold-start profile has 1 J/m^3 in one dx=1/8 cell, hence 1/8 J.
initial_energy = 0.125
rtol = 2.0e-6 if diffusion.dtype == np.float32 else 3.0e-13
np.testing.assert_allclose(np.sum(diffusion), initial_energy, rtol=rtol)
np.testing.assert_allclose(radiation[:, 2], initial_energy, rtol=rtol)
np.testing.assert_allclose(radiation[:, 4], initial_energy, rtol=rtol)
np.testing.assert_allclose(radiation[:, 3], 0.0, atol=1.0e-30)
np.testing.assert_allclose(radiation[:, 5:17], 0.0, atol=1.0e-30)
np.testing.assert_allclose(material, 0.0, atol=1.0e-30)

# Constant-opacity, periodic FLD must spread symmetrically from cell 3 while
# leaving distant cells untouched after one nearest-neighbor stencil update.
assert 0.0 < diffusion[3] < initial_energy
assert diffusion[2] > 0.0
assert diffusion[4] > 0.0
np.testing.assert_allclose(diffusion[2], diffusion[4], rtol=rtol)
np.testing.assert_allclose(diffusion[[0, 1, 5, 6, 7]], 0.0, atol=1.0e-30)

print(
    "transport-only FLD: "
    f"center={diffusion[3]:.16e} J, "
    f"neighbor={diffusion[2]:.16e} J, "
    f"total={np.sum(diffusion):.16e} J"
)
