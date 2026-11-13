"""Integrate signed plotfile charge on each side of a coarse/fine face."""

import sys

import numpy as np
import yt
from scipy.constants import elementary_charge


def charge_fractions(offset, radial_spacing=None):
    """Center the native cubic nodal distribution and integrate its two halves."""
    nodes = np.arange(-6, 7)
    distance = abs(nodes - offset)
    shape = np.where(
        distance < 1,
        2.0 / 3 - distance**2 + 0.5 * distance**3,
        np.maximum(2 - distance, 0) ** 3 / 6,
    )
    if radial_spacing is None:
        filtered = np.convolve(shape, [0.25, 0.5, 0.25], mode="same")
        cells = 0.5 * (filtered[:-1] + filtered[1:])
        on_parent = nodes[:-1] < 0
    else:
        # The radial face is at 32 mm. The RZ case filters only along z.
        radius = 0.032 + radial_spacing * nodes
        density = shape / radius
        cells = 0.5 * (density[:-1] + density[1:]) * (radius[:-1] + radial_spacing / 2)
        on_parent = nodes[:-1] >= 0
    return np.array([cells[on_parent].sum(), cells[~on_parent].sum()])


data = yt.load(sys.argv[1]).all_data()
geometry = sys.argv[2]
if geometry == "3d":
    volume = data["index", "cell_volume"].to_value("m**3")
else:
    # The AMReX frontend exposes the two native plotfile axes as x and y.
    volume = data["index", "dx"].to_value("m") * data["index", "dy"].to_value("m")
    if geometry == "rz":
        volume *= 2 * np.pi * data["index", "x"].to_value("m")
level = data["index", "grid_level"].to_ndarray()
# The offsets from the face are expressed in each particle's native cells.
if geometry == "rz":
    ions = charge_fractions(0.1, 0.002)
    electrons = -0.75 * charge_fractions(-0.2, 0.001)
else:
    ions = charge_fractions(-0.1)
    electrons = -0.75 * charge_fractions(0.2)
for field, expected in (
    ("rho_ions", ions),
    ("rho_electrons", electrons),
    ("rho", ions + electrons),
):
    charge = data["boxlib", field].to_ndarray() * volume / (1.0e9 * elementary_charge)
    measured = [charge[level == lev].sum() for lev in (0, 1)]
    np.testing.assert_allclose(measured, expected, rtol=2e-6, atol=2e-7)
