# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL

import pathlib
import sys

import pytest
from helpers import N_AXES, make_sim

import pywarpx

# make_sim, which writes the inputs file loaded below, is Cartesian only
pytestmark = pytest.mark.skipif(
    pywarpx.libwarpx.geometry_dim not in ("1d", "2d", "3d"),
    reason="make_sim does not build a non-Cartesian geometry yet",
)


def load_inputs_file(monkeypatch, argv, filename="inputs_test"):
    """Load an inputs file with 16 cells per axis in a process started as argv.

    Returns the geometry that WarpX built from it.
    """
    make_sim(n_cell=[16] * N_AXES[pywarpx.libwarpx.geometry_dim]).write_input_file(
        filename
    )

    monkeypatch.setattr(sys, "argv", argv)
    pywarpx.warpx.load_inputs_file(filename)

    return pywarpx.libwarpx.warpx.Geom(0)


def test_inputs_file_is_read_in_jupyter(monkeypatch):
    """AMReX accepts an inputs file as argv[1] only, and skips its parser
    entirely when argv[1] starts with "-". A Jupyter kernel runs as
    ``ipykernel_launcher.py -f kernel-1234.json``, so passing its sys.argv on
    would drop the inputs file and with it every parameter WarpX reads.
    """
    geom = load_inputs_file(
        monkeypatch, ["ipykernel_launcher.py", "-f", "kernel-1234.json"]
    )

    assert list(geom.domain.size) == [16] * N_AXES[pywarpx.libwarpx.geometry_dim]


def test_overrides_span_several_arguments(monkeypatch):
    """``python run.py -v amr.n_cell=8 8 geometry.prob_lo= -2.e-3 -2.e-3``
    overrides parameters of the inputs file as it does for the WarpX executable:
    the shell splits a value at its blanks, also in front of a negative number,
    and ``-v`` is an argument of run.py itself, which can precede the overrides.
    """
    n_axes = N_AXES[pywarpx.libwarpx.geometry_dim]
    geom = load_inputs_file(
        monkeypatch,
        ["run.py", "-v", "amr.n_cell=8"]
        + ["8"] * (n_axes - 1)
        + ["geometry.prob_lo="]
        + ["-2.e-3"] * n_axes,
    )

    assert list(geom.domain.size) == [8] * n_axes
    assert list(geom.ProbLo()) == pytest.approx([-2.0e-3] * n_axes)


def test_inputs_file_path_with_equals_sign(monkeypatch):
    """AMReX reads a first argument that contains "=" as a parameter instead of
    an inputs file, which a parameter scan with directories like ``a0=2/`` hits.
    """
    scan_dir = pathlib.Path("a0=2")
    scan_dir.mkdir()
    geom = load_inputs_file(monkeypatch, ["run.py"], scan_dir / "inputs_test")

    assert list(geom.domain.size) == [16] * N_AXES[pywarpx.libwarpx.geometry_dim]


@pytest.mark.parametrize("override_first", [True, False])
def test_double_dash_ends_the_overrides(monkeypatch, override_first):
    """AMReX parses no argument after "--", which leaves them to the script
    itself and, in a PETSc build, to PETSc, which ``amrex::Initialize`` starts
    with the same arguments. ``python run.py amr.n_cell=8 8 -- amr.n_cell=32 32``
    keeps 8 cells, and without the first override, the file's 16 cells.
    """
    n_axes = N_AXES[pywarpx.libwarpx.geometry_dim]
    before = ["amr.n_cell=8"] + ["8"] * (n_axes - 1) if override_first else []
    after = ["--", "amr.n_cell=32"] + ["32"] * (n_axes - 1)

    geom = load_inputs_file(monkeypatch, ["run.py"] + before + after)

    assert list(geom.domain.size) == [8 if override_first else 16] * n_axes
