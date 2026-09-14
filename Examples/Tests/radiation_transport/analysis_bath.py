#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL

"""Check the entire two-band drive/cooling trajectory and independent ledgers."""

import argparse
import re
from pathlib import Path

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from scipy.integrate import quad

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path)
parser.add_argument("--dims", type=int, choices=(1, 3), default=1)
parser.add_argument("--temperature", action="store_true")
add_precision_arguments(parser)
args = parser.parse_args()
field_dtype, _, _ = precision_dtypes(args)
path = Path("diags/radiation_energy.txt")
labels = re.findall(r"\[\d+\]([^\s]+)", path.read_text().splitlines()[0])
data = np.atleast_2d(np.loadtxt(path))
columns = {label: i for i, label in enumerate(labels)}
assert data.shape[1] == len(labels)
assert np.all(np.isfinite(data))

dt = 1e-10
conductance = 299792458.0 / (2.0 + 1.5 * 20.0)
energies = np.zeros(2)
injected = escaped = 0.0
expected = [(energies.copy(), injected, escaped)]
thermal_baths = []
for temperature in (10000.0, 5000.0):
    edge = 1e-19 / (1.380649e-23 * temperature)
    fraction = quad(lambda x: x**3 / np.expm1(x), 0.0, edge)[0] * 15 / np.pi**4
    thermal_baths.append(
        7.565733250280007e-16 * temperature**4 * np.array([fraction, 1 - fraction])
    )
for step in range(100):
    time = (step + 0.5) * dt
    bath = np.array(
        [
            [2.0] * (2 * (args.dims - 1)) + [1.0, 3.0],
            [3.5] * (2 * (args.dims - 1)) + [4.0, 3.0],
        ]
    )
    bath[0] *= time < 5e-9
    bath[1] *= time < 3e-9
    if args.temperature:
        assert args.dims == 1
        bath = np.column_stack(
            (thermal_baths[0] * (time < 5e-9), thermal_baths[1] * (time < 3e-9))
        )
    outward = dt * conductance * (energies[:, None] - bath)
    escaped += np.maximum(outward, 0.0).sum()
    injected += np.maximum(-outward, 0.0).sum()
    energies -= outward.sum(axis=1)
    expected.append((energies.copy(), injected, escaped))

# SP tolerance covers accumulation across 100 steps; DP checks algebra tightly.
rtol = 2e-5 if field_dtype == np.float32 else 2e-12
for row in data:
    step = round(row[columns["time(s)"]] / dt)
    energy, incoming, outgoing = expected[step]
    for group in range(2):
        np.testing.assert_allclose(
            row[columns[f"diffusion_radiation_group_{group}(J)"]],
            energy[group],
            rtol=rtol,
            atol=1e-30,
        )
    np.testing.assert_allclose(
        row[columns["cumulative_boundary_energy_injection(J)"]],
        incoming,
        rtol=rtol,
        atol=1e-30,
    )
    np.testing.assert_allclose(
        row[columns["cumulative_boundary_energy_loss(J)"]],
        outgoing,
        rtol=rtol,
        atol=1e-30,
    )
    residual = row[columns["total_radiation(J)"]] + outgoing - incoming
    assert abs(residual) <= rtol * max(incoming, 1e-30)
assert data[-1, columns["time(s)"]] > 9.99e-9
assert escaped > 0.0 and injected > escaped

if args.reference:
    reference = np.atleast_2d(np.loadtxt(args.reference / "diags/radiation_energy.txt"))
    np.testing.assert_allclose(data[-1], reference[-1], rtol=rtol, atol=1e-30)
print("Two-group driven-bath trajectory, injection, escape and restart ledgers pass.")
