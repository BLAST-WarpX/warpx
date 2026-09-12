#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Audit actual caloric energy and restart for stationary native hybrid LTE/FLD."""

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


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path)
parser.add_argument("--solver-details", action="store_true")
args = parser.parse_args()
paths = sorted(
    p
    for p in Path("diags").glob("diag1[0-9]*")
    if p.name.removeprefix("diag1").isdigit()
)
assert paths and paths[-1].name == "diag1000020"
if args.reference:
    reference = read(args.reference / "diags" / paths[-1].name)
    actual = read(paths[-1])
    for name, field in reference.items():
        np.testing.assert_allclose(actual[name], field, rtol=1e-11, atol=0)
else:
    assert len(paths) == 5
    totals = []
    temperatures = []
    for path in paths:
        fields = read(path)
        rho = np.squeeze(fields["rho"])
        te = np.squeeze(fields["Te"])
        radiation = np.squeeze(fields["radiation_diffusion_energy"])
        # Constant deposited density makes the average nodal rho*Te equal
        # rho times the plotted average Te; Pe is not a caloric proxy.
        np.testing.assert_allclose(rho, rho.mean(), rtol=3e-12)
        assert np.all(np.isfinite(te)) and np.all(te > 0)
        assert np.all(np.isfinite(radiation)) and np.all(radiation >= 0)
        material = 1.5 * Boltzmann * rho / elementary_charge * te / te.size
        totals.append(material.sum() + radiation.sum())
        temperatures.append(te)
    np.testing.assert_allclose(totals, totals[0], rtol=1e-10)
    assert np.max(np.abs(temperatures[-1] - temperatures[0])) > 1
if args.solver_details:
    ledger = Path("diags/radiation_energy.txt")
    with ledger.open() as stream:
        columns = re.findall(r"\[\d+\]([^\s]+)", stream.readline())
    values = np.atleast_2d(np.loadtxt(ledger))
    assert len(columns) == values.shape[1]
    data = dict(zip(columns, values.T))
    assert np.all(np.isfinite(values))
    steps = data["accepted_radiation_substeps()"]
    rejected = data["rejected_radiation_attempts()"]
    assert np.any(steps > 1) and np.any(rejected > 0)
    assert np.all(steps <= 64) and np.all(rejected <= 6)
    assert np.all(data["material_relative_residual()"] <= 1e-11)
    assert np.all(data["raw_stage_energy_relative_residual()"] <= 1e-10)
    assert np.all(data["minimum_group_cell_energy(J)"] >= 0)
    assert np.all(data["minimum_material_temperature(K)"] > 0)
    group_material = data["diffusion_group_0_cumulative_material(J)"]
    scale = np.max(data["total_radiation(J)"]) + np.max(np.abs(group_material))
    np.testing.assert_allclose(
        group_material,
        data["cumulative_material_exchange(J)"],
        rtol=0,
        atol=1e-10 * scale,
    )
    if not args.reference:
        np.testing.assert_allclose(
            group_material,
            np.cumsum(data["diffusion_group_0_material(J)"]),
            rtol=1e-13,
            atol=0,
        )
    else:
        reference_values = np.atleast_2d(np.loadtxt(args.reference / ledger))
        reference_data = dict(zip(columns, reference_values.T))
        for row, step in enumerate(data["step()"]):
            match = np.flatnonzero(reference_data["step()"] == step)
            assert match.size == 1
            for column in (
                "diffusion_group_0_cumulative_material(J)",
                "diffusion_group_0_cumulative_out(J)",
                "diffusion_group_0_cumulative_in(J)",
            ):
                np.testing.assert_allclose(
                    data[column][row],
                    reference_data[column][match[0]],
                    rtol=0,
                    atol=1e-10 * scale,
                )
print("Coupled native LTE/FLD caloric conservation and restart gates passed")
