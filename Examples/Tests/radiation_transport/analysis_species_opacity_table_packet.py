#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Validate mixed per-species spectral-table/parser opacity in RCYLINDER."""

from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer
from scipy.constants import c, elementary_charge


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
momentum = load_table(Path("diags/particle_momentum.txt"))
number = load_table(Path("diags/particle_number.txt"))

assert radiation.shape[0] == 2
assert radiation.shape[1] >= 17
assert particle.shape[0] == 2
assert particle.shape[1] >= 12
assert momentum.shape[0] == 2
assert momentum.shape[1] >= 5
assert number.shape[0] == 2

# At ni=1.25*n0, Te=100 eV and photon_energy=200 eV, the foam table
# interpolation gives 1050 m^-1 and the tungsten parser gives 4200 m^-1.
alpha_foam = 1050.0
alpha_tungsten = 4200.0
dt = 1.0e-12
transmission = np.exp(-(alpha_foam + alpha_tungsten) * c * dt)
packet_energy = 1.0e15 * 200.0 * elementary_charge

initial_streaming = radiation[0, 3]
final_streaming = radiation[-1, 3]
requested_material = initial_streaming - final_streaming
expected_initial = 4.0 * packet_energy
expected_final = 2.0 * packet_energy * (1.0 + transmission)
expected_absorbed = 2.0 * packet_energy * (1.0 - transmission)

material = final["radiation_material_energy"].squeeze()
d = np.sum(material)
single_precision = material.dtype == np.float32
analytic_rtol = 3.0e-5 if single_precision else 4.0e-12
conservation_rtol = 5.0e-5 if single_precision else 3.0e-11

np.testing.assert_allclose(initial_streaming, expected_initial, rtol=analytic_rtol)
np.testing.assert_allclose(final_streaming, expected_final, rtol=analytic_rtol)
np.testing.assert_allclose(requested_material, expected_absorbed, rtol=analytic_rtol)

# The two control packets prove that an absent named species contributes zero
# instead of clamping ni=0 to the positive lower table endpoint.
np.testing.assert_array_equal(np.flatnonzero(material), np.array([2]))
assert material[2] > 0.0
precision = np.float32 if single_precision else np.float64
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
np.testing.assert_allclose(radiation[:, 7:9], 0.0, atol=1.0e-24)
np.testing.assert_allclose(radiation[0, -2:], 0.0, rtol=0.0, atol=1.0e-24)
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

# ParticleEnergy columns are step, time, total, electrons, foam, tungsten,
# photons, followed by mean energies. The kinetic electron bath receives the
# complete packet loss and is independent of the radiation material ledger.
electron_gain = particle[-1, 3] - particle[0, 3]
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
np.testing.assert_allclose(particle[:, 6], radiation[:, 3], rtol=analytic_rtol)
electron_radiation_drift = electron_gain + radiation[-1, 2] - radiation[0, 2]
np.testing.assert_allclose(
    electron_radiation_drift,
    -residual,
    rtol=0.0,
    atol=particle_tolerance,
)

# Symmetric electron and photon pairs make the manufactured problem exactly
# momentum-free, and opacity lookup changes no charged-particle population.
momentum_scale = expected_initial / c
np.testing.assert_allclose(
    momentum[:, 2:5], 0.0, atol=conservation_rtol * momentum_scale
)
# ParticleNumber columns after step/time are total/individual macroparticle
# counts, total weight, then individual species weights. Electrons, foam and
# tungsten must be unchanged; only photon packet weight is attenuated.
initial_number = np.broadcast_to(number[0], number.shape)
np.testing.assert_array_equal(number[:, 3:6], initial_number[:, 3:6])
np.testing.assert_allclose(number[:, 8:11], initial_number[:, 8:11], rtol=analytic_rtol)
np.testing.assert_allclose(
    number[-1, 11] / number[0, 11], 0.5 * (1.0 + transmission), rtol=analytic_rtol
)

print(
    "RCYL mixed species spectral table/parser: "
    f"transmission={transmission:.16e}, "
    f"requested={requested_material:.16e} J, "
    f"realized={d:.16e} J, "
    f"residual={residual:.16e} J"
)
