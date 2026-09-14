# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL

import importlib.util
import os

import pytest

import pywarpx

# dimensionalities that pywarpx can load, see LibWarpX.load_library. Note that
# RCYLINDER and RSPHERE are valid WarpX_DIMS but are not selectable from Python
# yet, so add_warpx_pytest() does not register a test for them.
DIMS_MODULE = {
    "1": "warpx_pybind_1d",
    "2": "warpx_pybind_2d",
    "3": "warpx_pybind_3d",
    "RZ": "warpx_pybind_rz",
}

# which one a bare `pytest` run picks when several are available
DIMS_PREFERENCE = ("3", "2", "1", "RZ")

# tolerances for the conservation identities asserted by the unit tests
PRECISION_RTOL = {"SINGLE": 1.0e-5, "DOUBLE": 1.0e-12}


def available_dims():
    """Dimensionalities this pywarpx install was compiled with.

    Probing the module spec imports nothing, which matters because a
    ``warpx_pybind_*`` module can only be imported once per process.
    """
    return [
        dims
        for dims, module in DIMS_MODULE.items()
        if importlib.util.find_spec(f"pywarpx.{module}") is not None
    ]


def _select_dims():
    """The single dimensionality this pytest process runs.

    ``WARPX_TEST_DIMS`` is what ``add_warpx_pytest()`` sets, one ctest test per
    built dimensionality. A bare ``pytest`` run without it picks one of the
    built dimensionalities, so that the suite is equally usable outside ctest.
    """
    available = available_dims()
    if not available:
        raise pytest.UsageError(
            "No warpx_pybind_* module found. Build WarpX with -DWarpX_PYTHON=ON "
            "and make sure the package is importable, e.g. with "
            "`cmake --build build --target pip_install`."
        )

    requested = os.environ.get("WARPX_TEST_DIMS")
    if requested:
        if requested not in available:
            raise pytest.UsageError(
                f"WARPX_TEST_DIMS={requested} is not compiled into this install, "
                f"available: {available}."
            )
        return requested

    return next(dims for dims in DIMS_PREFERENCE if dims in available)


# Pin the process to one dimensionality here, at conftest import time, so that
# it is fixed before any test module is imported. That lets a test gate itself
# on the geometry with a plain skipif, and it makes Config available for the
# mpi4py decision below.
WARPX_DIMS = _select_dims()
pywarpx.geometry.dims = WARPX_DIMS
Config = pywarpx.libwarpx.libwarpx_so.Config

# mpi4py has to own MPI_Init. AMReX only calls MPI_Finalize in amrex::Finalize
# when it called MPI_Init itself (ParallelDescriptor::StartParallel checks
# MPI_Initialized), MPI cannot be initialized a second time once finalized, and
# every test here finalizes WarpX and builds a new simulation afterwards.
# Importing mpi4py keeps MPI alive for the whole pytest process.
if Config.have_mpi:
    from mpi4py import MPI

    assert MPI.Is_initialized()


def rtol():
    """Relative tolerance matching the precision WarpX was compiled with.

    Deposition combines particle attributes (``ParticleReal``) with field data
    (``Real``), so a mixed-precision build is held to the looser of the two.
    """
    return max(
        PRECISION_RTOL[Config.precision], PRECISION_RTOL[Config.precision_particles]
    )


# autouse: pytest wraps every test_* function under tests/unit in this, once per
# parametrized case, without the test having to ask for it. What is before the
# yield runs as setup, what is after it as teardown, also when the test fails.
@pytest.fixture(autouse=True, scope="function")
def warpx_lifecycle(tmp_path, monkeypatch):
    """Isolate each test and guarantee a clean WarpX/AMReX teardown.

    Each test runs in its own temporary directory, so that diagnostics and
    backtrace files never collide between tests. On teardown, WarpX and AMReX
    are finalized and all module-level input state is cleared, so that the next
    test starts from an empty input deck.

    This is what lets a single process run several independent simulations, for
    instance to compare deposition algorithms.
    """
    monkeypatch.chdir(tmp_path)

    yield

    pywarpx.warpx.finalize()
