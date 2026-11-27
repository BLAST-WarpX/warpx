#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Manufactured two-group native table/analytic LTE-FLD interface qualification."""

import argparse
import re
from pathlib import Path

import numpy as np
from read_raw_data import _read_buffer
from scipy.constants import Boltzmann, elementary_charge


def read(path):
    with (path / "Header").open() as header:
        header.readline()
        names = [header.readline().strip() for _ in range(int(header.readline()))]
    return _read_buffer(str(path), str(path / "Level_0/Cell_H"), names)


def ledger(path):
    with path.open() as stream:
        names = re.findall(r"\[\d+\]([^\s]+)", stream.readline())
    values = np.atleast_2d(np.loadtxt(path))
    assert len(names) == values.shape[1] and np.all(np.isfinite(values))
    return dict(zip(names, values.T))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path)
parser.add_argument("--restart", action="store_true")
args = parser.parse_args()
paths = sorted(
    path
    for path in Path("diags").glob("diag1[0-9]*")
    if path.name.removeprefix("diag1").isdigit()
)
assert paths and paths[-1].name == "diag1000020"
totals, temperatures, group_totals = [], [], []
for path in paths:
    fields = read(path)
    rho = np.squeeze(fields["rho"])
    te = np.squeeze(fields["Te"])
    np.testing.assert_allclose(rho, rho.mean(), rtol=3e-12)
    assert np.all(np.isfinite(te)) and np.all(te > 0)
    energy = []
    for group in range(2):
        radiation = np.squeeze(fields[f"radiation_diffusion_energy_g{group}"])
        assert np.all(np.isfinite(radiation)) and np.all(radiation >= 0)
        energy.append(radiation.sum())
    # Unit-length periodic 1D domain, native ideal caloric density, constant rho.
    material = 1.5 * Boltzmann * rho / elementary_charge * te / te.size
    totals.append(material.sum() + sum(energy))
    group_totals.append(energy)
    temperatures.append(te)
np.testing.assert_allclose(totals, totals[0], rtol=1e-10)
if not args.restart:
    assert len(paths) == 5
    assert np.mean(temperatures[-1] - temperatures[0]) > 1
    assert np.all(np.array(group_totals[-1]) < group_totals[0])

data = ledger(Path("diags/radiation_energy.txt"))
assert np.all(data["material_relative_residual()"] <= 1e-11)
assert np.all(data["raw_stage_energy_relative_residual()"] <= 1e-10)
assert np.all(data["minimum_group_cell_energy(J)"] >= 0)
assert np.all(data["minimum_material_temperature(K)"] > 0)
np.testing.assert_allclose(
    data["diffusion_group_0_cumulative_material(J)"]
    + data["diffusion_group_1_cumulative_material(J)"],
    data["cumulative_material_exchange(J)"],
    rtol=0,
    atol=1e-10 * totals[0],
)
if args.reference:
    reference = read(args.reference / "diags" / paths[-1].name)
    actual = read(paths[-1])
    for name in reference:
        scale = max(np.max(np.abs(reference[name])), 1e-100)
        np.testing.assert_allclose(
            actual[name], reference[name], rtol=1e-11, atol=1e-13 * scale
        )
    expected = ledger(args.reference / "diags/radiation_energy.txt")
    for row, step in enumerate(data["step()"]):
        match = np.flatnonzero(expected["step()"] == step)
        assert match.size == 1
        for group in range(2):
            for quantity in ("cumulative_material", "cumulative_out", "cumulative_in"):
                key = f"diffusion_group_{group}_{quantity}(J)"
                np.testing.assert_allclose(
                    data[key][row],
                    expected[key][match[0]],
                    rtol=0,
                    atol=1e-10 * totals[0],
                )
print(
    "Two-group native caloric conservation, material response and comparison gates passed"
)
