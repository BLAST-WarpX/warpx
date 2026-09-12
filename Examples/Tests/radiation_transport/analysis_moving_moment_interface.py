#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Actual moving-interface inventories; native nodal EOS, not cell-averaged T*rho."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from analysis_moving_moment_pulse import KB, MP, QE, C, plotfiles, state

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "Tools" / "PostProcessing")
)
from read_raw_data import _fab_real_dtype, _read_header  # noqa: E402


def nodes(header_path, cells, periodic=True):
    """Reuse the repository VisMF parser; read valid nodes, never stale ghosts."""
    header_path = Path(header_path)
    boxes, names, offsets, header = _read_header(str(header_path))
    assert header.ncomp == 1 and header.version in (1, 2)
    cells = np.atleast_1d(cells).astype(int)
    result = np.full(tuple(cells + 1), np.nan)
    ghosts = np.broadcast_to(np.asarray(header.nghost, dtype=int), cells.shape)
    for (lo, hi, kind), name, offset in zip(boxes, names, offsets, strict=True):
        assert len(lo) == len(cells) and np.all(np.asarray(kind) == 1)
        lo, hi = np.asarray(lo, dtype=int), np.asarray(hi, dtype=int)
        first, last = lo + ghosts, hi - ghosts
        assert np.all((0 <= first) & (first <= last) & (last <= cells))
        with (header_path.parent / name).open("rb") as stream:
            stream.seek(offset)
            dtype = (
                _fab_real_dtype(stream.readline())
                if header.version == 1
                else np.float64
            )
            values = np.fromfile(stream, dtype=dtype, count=int(np.prod(hi - lo + 1)))
        values = values.reshape(tuple(hi - lo + 1), order="F")
        valid = tuple(
            slice(int(g), int(g + b - a + 1))
            for g, a, b in zip(ghosts, first, last, strict=True)
        )
        destination = tuple(
            slice(int(a), int(b + 1)) for a, b in zip(first, last, strict=True)
        )
        values = values[valid]
        assert np.all(np.isfinite(values))
        previous = result[destination]
        shared = np.isfinite(previous)
        if np.any(shared):
            assert np.max(abs(previous[shared] - values[shared])) <= 64 * np.finfo(
                float
            ).eps * np.max(abs(values))
        result[destination] = values
    assert np.all(np.isfinite(result))
    periodic = np.broadcast_to(np.asarray(periodic, dtype=bool), cells.shape)
    for axis in range(len(cells)):
        if periodic[axis]:
            assert np.max(
                abs(np.take(result, 0, axis=axis) - np.take(result, -1, axis=axis))
            ) <= 64 * np.finfo(float).eps * np.max(abs(result))
    return np.asarray(
        result[
            tuple(
                slice(0, int(n) + (not p)) for n, p in zip(cells, periodic, strict=True)
            )
        ],
        dtype=np.longdouble,
    )


def load(plot):
    result = state(plot)
    step = int(plot.name[5:])
    result["rho_nodes"] = nodes(
        plot / "raw_fields" / "Level_0" / "rho_fp_H", result["n"]
    )
    checkpoint = plot.parent / f"chk{step:06d}" / "Level_0"
    result["temperature_nodes"] = nodes(
        checkpoint / "hybrid_electron_temperature_fp[level=0]_H", result["n"]
    )
    result["step"] = step
    return result


def caloric(row):
    ev = row["temperature_nodes"] * KB / QE
    return row["rho_nodes"] * (np.longdouble("1.5") * ev + 2 * ev**4 / (1 + ev**4))


def check(
    directory, reference_directory=None, compare_restart=False, require_laminar=False
):
    directory = Path(directory)
    plots = plotfiles(directory)
    originals = plots if reference_directory is None else plotfiles(reference_directory)
    initial = load(originals[0])
    assert initial["step"] == 0
    assert abs(state(plots[-1])["time"] / (1e-8 / 3) - 1) < 1e-12
    mass = initial["weight"] * MP
    radiation0 = np.sum(initial["radiation_diffusion_energy"])
    gamma0 = np.sqrt(1 + np.sum(initial["u"] ** 2, axis=1) / C**2)
    internal0 = caloric(initial)
    history = []
    for plot in plots:
        current = load(plot)
        assert np.array_equal(initial["ids"], current["ids"])
        assert np.array_equal(initial["weight"], current["weight"])
        gamma = np.sqrt(1 + np.sum(current["u"] ** 2, axis=1) / C**2)
        du = current["u"] - initial["u"]
        kinetic = np.sum(
            mass * np.sum(du * (current["u"] + initial["u"]), axis=1) / (gamma0 + gamma)
        )
        internal = np.sum(caloric(current) - internal0) * initial["dx"]
        carry = np.sum(mass * (current["carry_work"] - initial["carry_work"]))
        radiation = np.sum(
            current["radiation_diffusion_energy"]
            - initial["radiation_diffusion_energy"]
        )
        energy_error = radiation + kinetic + internal + carry
        momentum_error = []
        for d, axis in enumerate("xyz"):
            change = (
                np.sum(
                    current[f"radiation_moment_q{axis}"]
                    - initial[f"radiation_moment_q{axis}"]
                )
                / C
            )
            change += np.sum(
                mass * (du[:, d] + current["carry"][:, d] - initial["carry"][:, d])
            )
            momentum_error.append(float(change))
            bound = 1e-8 * radiation0 / C + 128 * np.finfo(float).eps * np.sum(
                mass * abs(initial["u"][:, d])
            )
            assert abs(change) < bound, (current["step"], change, bound)
        assert abs(energy_error) < 1e-8 * radiation0, (
            current["step"],
            energy_error,
            radiation0,
        )
        # Test whether the cold ion map remains laminar. Do not assume a FLASH
        # single-fluid closure once different ion streams cross.
        old_order = np.argsort(initial["position"])
        new_order = np.argsort(current["position"])
        shift = int(np.flatnonzero(new_order == old_order[0])[0])
        laminar = bool(np.array_equal(np.roll(new_order, -shift), old_order))
        ev = current["temperature_nodes"] * KB / QE
        extra_cv = 8 * ev**3 / (1 + ev**4) ** 2
        row = {
            "step": current["step"],
            "time_s": current["time"],
            "radiation_change_J": float(radiation),
            "kinetic_change_J": float(kinetic),
            "electron_caloric_change_J": float(internal),
            "carry_change_J": float(carry),
            "raw_energy_error_J": float(energy_error),
            "energy_error_over_initial_radiation": float(
                abs(energy_error) / radiation0
            ),
            "raw_momentum_error_kg_m_s": momentum_error,
            "laminar_particle_ordering": laminar,
            "latent_heat_capacity_fraction_max": float(np.max(extra_cv / 1.5)),
            "minimum_temperature_eV": float(np.min(ev)),
            "velocity_max_relative_change": float(
                np.max(abs(current["u"][:, 2] / gamma / 9e4 - 1))
            ),
        }
        history.append(row)
    assert abs(history[-1]["kinetic_change_J"]) / radiation0 > 1e-3
    assert history[-1]["velocity_max_relative_change"] > 0.01
    ledger = np.loadtxt(directory / "diags" / "radiation_energy.txt")
    assert (
        np.max(abs(ledger[:, 2] + ledger[:, 6] + ledger[:, 16] - radiation0))
        < 1e-10 * radiation0
    )
    report = {
        "cells": initial["n"],
        "initial_radiation_J": float(radiation0),
        "history": history,
    }
    if compare_restart:
        assert reference_directory is not None
        reference = load(originals[-1])
        assert current["step"] == reference["step"]
        assert np.array_equal(current["ids"], reference["ids"])
        assert np.array_equal(current["weight"], reference["weight"])
        length = initial["n"] * initial["dx"]
        displacement = current["position"] - reference["position"]
        displacement = (displacement + length / 2) % length - length / 2
        assert np.max(abs(displacement)) < 1e-11 * length
        assert np.max(abs(current["u"] - reference["u"])) < 1e-12 * 9e4
        for name in (
            "radiation_diffusion_energy",
            "radiation_moment_qx",
            "radiation_moment_qy",
            "radiation_moment_qz",
            "rho_nodes",
            "temperature_nodes",
        ):
            scale = np.max(abs(reference[name]))
            assert np.max(abs(current[name] - reference[name])) <= 1e-10 * scale, name
        report["changed_rank_restart_matches"] = True
    (directory / "interface_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if require_laminar:
        assert all(row["laminar_particle_ordering"] for row in history), (
            "The cold moving-interface trajectory develops particle crossings; "
            "inventory conservation alone does not qualify this benchmark."
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--reference")
    parser.add_argument("--compare-restart", action="store_true")
    parser.add_argument("--require-laminar", action="store_true")
    args = parser.parse_args()
    check(args.directory, args.reference, args.compare_restart, args.require_laminar)
