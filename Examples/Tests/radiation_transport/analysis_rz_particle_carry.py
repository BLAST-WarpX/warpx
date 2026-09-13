#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL

"""Native RZ absorption/work/carry ledgers over a finite absorption interval.

This is a lab-frame opacity test, not a boosted-equilibrium or diffusion test.
"""

from pathlib import Path

import numpy as np


def read_table(name):
    path = Path("diags") / name
    with path.open() as stream:
        labels = [token.split("]", 1)[1] for token in stream.readline().split()]
    data = np.atleast_2d(np.loadtxt(path))
    assert data.shape == (257, len(labels))
    assert np.all(np.isfinite(data))
    return {key: data[:, i] for i, key in enumerate(labels)}


energy = read_table("radiation_energy.txt")
momentum = read_table("radiation_momentum.txt")
time = energy["time(s)"]
np.testing.assert_allclose(time, np.arange(257) * 1.0e-12, rtol=2.0e-12, atol=0)
initial = energy["total_radiation(J)"][0]
expected = initial * np.exp(-10.0 * 299792458.0 * time)
np.testing.assert_allclose(energy["total_radiation(J)"], expected, rtol=2.0e-12)
assert 0.5 < (initial - expected[-1]) / initial < 0.6

pending = energy["pending_material_carry_energy(J)"]
balance = (
    energy["total_radiation(J)"]
    + energy["cumulative_material_exchange(J)"]
    + energy["cumulative_boundary_energy_loss(J)"]
    + energy["cumulative_numerical_energy_residual(J)"]
    + pending
)
np.testing.assert_allclose(balance, initial, rtol=2.0e-12)
np.testing.assert_allclose(
    energy["material_exchange(J)"],
    energy["material_internal_exchange(J)"] + energy["material_kinetic_exchange(J)"],
    rtol=2.0e-12,
    atol=2.0e-12 * initial,
)
assert np.sum(np.abs(energy["material_kinetic_exchange(J)"])) > 1.0e-10 * initial
np.testing.assert_allclose(
    energy["cumulative_boundary_energy_loss(J)"], 0, atol=1.0e-18
)

# The ray remains on +r without escape; particle carry is projected locally.
requested = (initial - expected) / 299792458.0
radial = momentum["cumulative_material_r(kg*m/s)"]
radial = radial + momentum["pending_streaming_material_r(kg*m/s)"]
np.testing.assert_allclose(
    radial, requested, rtol=2.0e-12, atol=2.0e-12 * requested[-1]
)
print(
    f"256-step RZ absorption: {initial - expected[-1]:.9g} J transferred; ledgers pass"
)
