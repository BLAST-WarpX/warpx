# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL

import pytest

import pywarpx

# Import mpi4py before AMReX is initialized for the first time.
#
# This is what makes several simulations per process possible: AMReX only calls
# MPI_Finalize in amrex::Finalize if it called MPI_Init itself
# (ParallelDescriptor::StartParallel checks MPI_Initialized). Once MPI has been
# finalized it cannot be initialized again, so without mpi4py owning MPI_Init,
# the second simulation in a process aborts with "Attempting to use an MPI
# routine ... after finalizing".
#
# The dimensionality, and hence Config.have_mpi, is not known at collection
# time, so this cannot be conditioned on the build having MPI. Importing mpi4py
# is harmless for a serial build.
try:
    from mpi4py import MPI

    mpi_owned_by_mpi4py = MPI.Is_initialized()
except ImportError:
    # mpi4py is optional. Without it, AMReX initializes and finalizes MPI
    # itself, which limits a process to a single simulation.
    mpi_owned_by_mpi4py = False


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "manages_warpx: test drives the WarpX/AMReX init and finalize itself",
    )


@pytest.fixture(autouse=True, scope="function")
def warpx_lifecycle(request, tmp_path, monkeypatch):
    """Isolate each test and guarantee a clean WarpX/AMReX teardown.

    Each test runs in its own temporary directory, so that diagnostics and
    backtrace files never collide between tests. On teardown, WarpX and AMReX
    are finalized and all module-level input state is cleared, so that the next
    test starts from an empty input deck.

    Unlike the equivalent fixtures in ImpactX and pyAMReX, this cannot
    initialize AMReX up front: WarpX reads its whole input deck during
    initialization, so a test has to assemble that deck first. This fixture
    therefore owns the teardown half only.
    """
    monkeypatch.chdir(tmp_path)

    yield

    if not request.node.get_closest_marker("manages_warpx"):
        pywarpx.warpx.finalize()
