# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL

import pathlib

import pytest

import pywarpx

# directory name -> geometry.dims, the same mapping that add_warpx_pytest()
# uses to turn a subdirectory into a ctest test
DIMS_OF_DIRECTORY = {
    "1d": "1",
    "2d": "2",
    "3d": "3",
    "rz": "RZ",
    "rcylinder": "RCYLINDER",
    "rsphere": "RSPHERE",
}

_SUITE_ROOT = pathlib.Path(__file__).parent


def _dims_of(path):
    """Return the dimensionality of the subdirectory a test file lives in."""
    for parent in path.parents:
        if parent == _SUITE_ROOT:
            break
        dims = DIMS_OF_DIRECTORY.get(parent.name)
        if dims is not None:
            return dims
    return None


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "manages_warpx: test drives the WarpX/AMReX init and finalize itself",
    )


def pytest_collection_modifyitems(session, config, items):
    """Pin this process to one dimensionality and let mpi4py own ``MPI_Init``.

    Which ``warpx_pybind_*`` module gets imported is decided by
    ``geometry.dims``, and it can only be imported once per process. Setting it
    here, instead of leaving it to the first PICMI grid that a test constructs,
    makes ``Config`` available while the process is still empty. That is what
    allows the ``mpi4py`` import below to be conditional, as it is in ImpactX,
    rather than unconditional and wrapped in a bare ``except ImportError``.

    The ``mpi4py`` import has to happen before the first ``amrex::Initialize``.
    AMReX only calls ``MPI_Finalize`` in ``amrex::Finalize`` when it called
    ``MPI_Init`` itself (``ParallelDescriptor::StartParallel`` checks
    ``MPI_Initialized``), and MPI cannot be initialized a second time once it
    has been finalized. Importing ``mpi4py`` first therefore hands ``MPI_Init``
    to mpi4py, which keeps MPI alive for the whole pytest process, so that
    every test can finalize WarpX and build a new simulation afterwards.
    """
    dims = {_dims_of(item.path) for item in items}
    dims.discard(None)
    if not dims:
        return

    if len(dims) > 1:
        raise pytest.UsageError(
            "Cannot test several dimensionalities in one pytest process, got "
            f"{sorted(dims)}. A compiled warpx_pybind_* module can only be "
            "imported once per process, so run one subdirectory at a time, "
            "which is also how ctest invokes this suite."
        )

    pywarpx.geometry.dims = dims.pop()
    try:
        warpx_config = pywarpx.libwarpx.libwarpx_so.Config
    except Exception as exc:
        for item in items:
            item.add_marker(pytest.mark.skip(reason=str(exc)))
        return

    if warpx_config.have_mpi:
        from mpi4py import MPI

        # importing mpi4py.MPI initializes MPI and, because the module stays in
        # sys.modules, keeps it initialized for the rest of the process
        assert MPI.Is_initialized()


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
