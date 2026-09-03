#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Validate additive species opacity with kinetic electrons in RCYLINDER."""

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


final = load_plotfile(Path("diags/diag1000001"))
radiation = load_table(Path("diags/radiation_energy.txt"))
particle = load_table(Path("diags/particle_energy.txt"))

assert radiation.shape[0] == 2
assert radiation.shape[1] >= 17
assert particle.shape[0] == 2
# step, time, total, electrons, foam, tungsten, photons, then mean energies
assert particle.shape[1] >= 12

dt = 1.0e-12
alpha_foam = 1.0e3
alpha_tungsten = 4.0 * alpha_foam
expected_transmission = np.exp(-(alpha_foam + alpha_tungsten) * c * dt)

initial_streaming = radiation[0, 3]
final_streaming = radiation[-1, 3]
requested_material = initial_streaming - final_streaming
material = final["radiation_material_energy"].squeeze()
d = np.sum(material)
single_precision = material.dtype == np.float32
rtol = 2.0e-5 if single_precision else 3.0e-12
precision = np.float32 if single_precision else np.float64

assert initial_streaming > 0.0
assert requested_material > 0.0
np.testing.assert_allclose(
    final_streaming / initial_streaming, expected_transmission, rtol=rtol
)
np.testing.assert_allclose(
    requested_material,
    initial_streaming * (1.0 - expected_transmission),
    rtol=rtol,
)

# Only cell 2 contains both opacity species and the packet. Its material gain
# and the material ledgers must carry the realized transfer d.
np.testing.assert_array_equal(np.flatnonzero(material), np.array([2]))
assert material[2] > 0.0
field_roundoff = (
    32.0
    * np.finfo(precision).eps
    * max(
        abs(radiation[0, 2]),
        abs(radiation[-1, 2]),
        abs(requested_material),
        abs(d),
        abs(radiation[-1, 5]),
        abs(radiation[-1, 6]),
    )
)
field_tolerance = max(1.0e-24, field_roundoff)
np.testing.assert_allclose(radiation[-1, 5], d, rtol=0.0, atol=field_tolerance)
np.testing.assert_allclose(radiation[-1, 6], d, rtol=0.0, atol=field_tolerance)
np.testing.assert_allclose(radiation[-1, 9], d, rtol=0.0, atol=field_tolerance)
residual = radiation[-1, -2]
cumulative_residual = radiation[-1, -1]
np.testing.assert_allclose(
    residual,
    requested_material - d,
    rtol=0.0,
    atol=field_tolerance,
)
np.testing.assert_allclose(
    residual,
    radiation[0, 2] - radiation[-1, 2] - d,
    rtol=0.0,
    atol=field_tolerance,
)
np.testing.assert_allclose(
    cumulative_residual, residual, rtol=0.0, atol=field_tolerance
)
np.testing.assert_allclose(radiation[0, -2:], 0.0, rtol=0.0, atol=1.0e-24)
electron_gain = particle[-1, 3] - particle[0, 3]
# The kinetic diagnostic sums finite-macroparticle energies, so its reduction
# accumulates more floating-point error than the independently exact
# radiation/material field ledgers checked above.
particle_roundoff = (
    32.0
    * np.finfo(precision).eps
    * max(
        abs(particle[0, 3]),
        abs(particle[-1, 3]),
        abs(particle[0, 6]),
        abs(particle[-1, 6]),
        abs(radiation[0, 2]),
        abs(radiation[-1, 2]),
        abs(d),
    )
)
particle_tolerance = max(1.0e-24, particle_roundoff)
np.testing.assert_allclose(electron_gain, d, rtol=0.0, atol=particle_tolerance)
electron_radiation_drift = electron_gain + radiation[-1, 2] - radiation[0, 2]
np.testing.assert_allclose(
    electron_radiation_drift,
    -residual,
    rtol=0.0,
    atol=particle_tolerance,
)
np.testing.assert_allclose(radiation[:, 7:9], 0.0, atol=1.0e-24)

print(
    "RCYL additive species opacity: "
    f"transmission={final_streaming / initial_streaming:.16e}, "
    f"expected={expected_transmission:.16e}, "
    f"requested={requested_material:.16e} J, "
    f"realized={d:.16e} J, "
    f"residual={residual:.16e} J"
)
