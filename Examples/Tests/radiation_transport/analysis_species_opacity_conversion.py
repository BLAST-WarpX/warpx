#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Validate additive species Rosseland opacity and packet conversion."""

from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer
from scipy.constants import c


def load_plotfile(path: Path) -> dict[str, np.ndarray]:
    with open(path / "Header") as header:
        header.readline()
        n_fields = int(header.readline())
        field_names = [header.readline().strip() for _ in range(n_fields)]
    return _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), field_names)


def load_table(path: Path) -> np.ndarray:
    table = np.loadtxt(path)
    return table[np.newaxis, :] if table.ndim == 1 else table


radiation = load_table(Path("diags/radiation_energy.txt"))
final = load_plotfile(Path("diags/diag1000001"))

assert radiation.shape == (2, 17)
initial_total = radiation[0, 2]
initial_streaming = radiation[0, 3]
final_total = radiation[-1, 2]
final_streaming = radiation[-1, 3]
final_diffusion = radiation[-1, 4]
diffusion_cells = final["radiation_diffusion_energy"].squeeze()

assert initial_total > 0.0
np.testing.assert_allclose(initial_streaming, initial_total, rtol=2.0e-14)

# Each configured partial depth is thin: 400*dr=0.4 and 700*dr=0.7.
# Their sum is thick: (400+700)*dr=1.1. Exact disappearance of the original
# packet therefore proves that the two Rosseland contributions were added
# before applying the conversion criterion.
assert 400.0 * 1.0e-3 < 1.0
assert 700.0 * 1.0e-3 < 1.0
assert (400.0 + 700.0) * 1.0e-3 >= 1.0
np.testing.assert_allclose(final_streaming, 0.0, atol=1.0e-24)
assert final_diffusion > 0.0
np.testing.assert_allclose(final_diffusion, initial_total, rtol=2.0e-13)
np.testing.assert_allclose(final_total, initial_total, rtol=2.0e-13)

# With dt=1e-13 s the explicit FLD update takes exactly one substep. Starting
# from cell 2, its two cylindrical face transfers give a quantitative check
# of the summed alpha_R=1100 m^-1, including the representative photon-energy
# argument used by the species parsers.
dr = 1.0e-3
dt = 1.0e-13
alpha_total = 400.0 + 700.0
source_volume = 5.0 * np.pi * dr**2
source_density = initial_total / source_volume
dimensionless_gradient = 2.0 / (alpha_total * dr)
flux_limiter = (2.0 + dimensionless_gradient) / (
    6.0 + 3.0 * dimensionless_gradient + dimensionless_gradient**2
)
diffusion_coefficient = c * flux_limiter / alpha_total
inner_transfer = (
    diffusion_coefficient * (2.0 * np.pi * 2.0 * dr) * source_density / dr * dt
)
outer_transfer = (
    diffusion_coefficient * (2.0 * np.pi * 3.0 * dr) * source_density / dr * dt
)
expected_cells = np.array(
    [
        0.0,
        inner_transfer,
        initial_total - inner_transfer - outer_transfer,
        outer_transfer,
    ]
)
cell_rtol = 3.0e-5 if diffusion_cells.dtype == np.float32 else 3.0e-12
np.testing.assert_allclose(diffusion_cells, expected_cells, rtol=cell_rtol)
np.testing.assert_allclose(np.sum(diffusion_cells), initial_total, rtol=cell_rtol)

# Zero Planck and absorption coefficients plus reflecting boundaries leave no
# material or escape channel. Conversion changes only the representation.
np.testing.assert_allclose(radiation[:, 5:17], 0.0, atol=1.0e-24)

print(
    "RCYL additive species Rosseland opacity: "
    f"streaming={final_streaming:.16e} J, "
    f"diffusion={final_diffusion:.16e} J, "
    f"inner transfer={diffusion_cells[1]:.16e} J, "
    f"total residual={final_total - initial_total:.6e} J"
)
