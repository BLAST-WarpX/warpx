# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL

import importlib.util
import os

import numpy as np
import pytest

import pywarpx
from pywarpx import picmi

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

# PICMI grid per dimensionality, and how many grid axes it takes
GRID_CLASS = {
    "1": picmi.Cartesian1DGrid,
    "2": picmi.Cartesian2DGrid,
    "3": picmi.Cartesian3DGrid,
    "RZ": picmi.CylindricalGrid,
}
N_AXES = {"1": 1, "2": 2, "3": 3, "RZ": 2}

# tolerances for the conservation identities asserted by the unit tests
RTOL = {"SINGLE": 1.0e-5, "DOUBLE": 1.0e-12}


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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "manages_warpx: test drives the WarpX/AMReX init and finalize itself",
    )


def pytest_report_header(config):
    return f"warpx: {WARPX_DIMS}D geometry, built: {available_dims()}"


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


@pytest.fixture(scope="function")
def dims():
    """The dimensionality of this process, as in ``geometry.dims``."""
    return WARPX_DIMS


@pytest.fixture(scope="function")
def make_sim():
    """Factory for a minimal, initialized simulation of this dimensionality.

    Returns a callable so that a test can create a simulation with the exact
    numerics it wants to exercise. Only one simulation may live at a time; the
    ``warpx_lifecycle`` fixture tears it down again after each test, which is
    what makes parametrizing over e.g. the current deposition algorithm
    possible.
    """

    def _make(
        n_cell=None,
        lower_bound=None,
        upper_bound=None,
        max_grid_size=8,
        particle_shape="quadratic",
        current_deposition_algo=None,
        dt=None,
    ):
        if WARPX_DIMS not in ("1", "2", "3"):
            raise NotImplementedError(
                f"make_sim does not build a {WARPX_DIMS} geometry yet. Tests "
                "that rely on it should skip on a non-Cartesian geometry."
            )

        n_axes = N_AXES[WARPX_DIMS]
        n_cell = [16] * n_axes if n_cell is None else list(n_cell)
        lower_bound = [-1.0e-3] * n_axes if lower_bound is None else list(lower_bound)
        upper_bound = [1.0e-3] * n_axes if upper_bound is None else list(upper_bound)

        # dt=None lets WarpX pick the CFL-limited time step, which keeps the
        # per-step particle displacement a sizeable fraction of a cell
        grid = GRID_CLASS[WARPX_DIMS](
            number_of_cells=n_cell,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            lower_boundary_conditions=["periodic"] * n_axes,
            upper_boundary_conditions=["periodic"] * n_axes,
            lower_boundary_conditions_particles=["periodic"] * n_axes,
            upper_boundary_conditions_particles=["periodic"] * n_axes,
            # more than one box, so that the guard cell exchange is exercised
            warpx_max_grid_size=max_grid_size,
        )
        solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)
        electrons = picmi.Species(particle_type="electron", name="electrons")

        sim = picmi.Simulation(
            solver=solver,
            time_step_size=dt,
            max_steps=1,
            verbose=0,
            particle_shape=particle_shape,
            warpx_current_deposition_algo=current_deposition_algo,
        )
        # no particles are injected: the tests add them explicitly
        sim.add_species(
            electrons,
            layout=picmi.GriddedLayout(
                n_macroparticle_per_cell=[0] * n_axes, grid=grid
            ),
        )

        sim.initialize_inputs()

        # AMReX runtime parameters, mirroring the ones ImpactX and pyAMReX use
        # in their pytest suites
        pywarpx.warpx.get_bucket("tiny_profiler").enabled = 0
        #   throw exceptions instead of writing Backtrace files, so a debugger
        #   can be attached
        pywarpx.amrex.throw_exception = 1
        pywarpx.amrex.signal_handling = 0
        #   abort GPU runs on out-of-memory instead of swapping to host RAM
        pywarpx.amrex.abort_on_out_of_gpu_memory = 1
        #   allocate GPU memory on demand instead of pre-allocating 3/4th, so
        #   that tests can share a GPU
        pywarpx.amrex.the_arena_init_size = 0

        sim.initialize_warpx()

        return sim

    return _make


def _rtol():
    """Relative tolerance matching the precision WarpX was compiled with.

    Deposition combines particle attributes (``ParticleReal``) with field data
    (``Real``), so a mixed-precision build is held to the looser of the two.
    """
    return max(RTOL[Config.precision], RTOL[Config.precision_particles])


def _uniform_particles(sim, n_per_dim=4, weight=1.0e6, ux=0.0, uy=0.0, uz=0.0):
    """Add a regular lattice of macro particles to the ``electrons`` species.

    One lattice point per grid axis of the current geometry, so this yields
    ``n_per_dim`` particles in 1D and ``n_per_dim**3`` in 3D. The particles sit
    strictly inside the domain and are offset from both cell centers and nodes,
    so that the shape factors of every deposition order spread charge over more
    than one cell.

    Returns the particle container together with the position and momentum
    arrays that were used, as they are needed to form the expected values.
    """
    geom = sim.extension.warpx.Geom(0)
    lo = np.array(geom.ProbLo())
    hi = np.array(geom.ProbHi())

    # fractional positions in (0, 1), avoiding the domain boundaries
    frac = (np.arange(n_per_dim) + 0.37) / n_per_dim
    axes = np.meshgrid(*(frac,) * len(lo), indexing="ij")
    coords = [(lo[i] + axis * (hi[i] - lo[i])).ravel() for i, axis in enumerate(axes)]

    n_part = coords[0].size
    zeros = np.zeros(n_part)
    if WARPX_DIMS == "1":
        x, y, z = zeros, zeros, coords[0]
    elif WARPX_DIMS == "2":
        x, y, z = coords[0], zeros, coords[1]
    else:
        x, y, z = coords

    w = np.full(n_part, weight)
    uxp = np.full(n_part, ux)
    uyp = np.full(n_part, uy)
    uzp = np.full(n_part, uz)

    electrons = sim.particles.get("electrons")
    electrons.add_particles(
        x=x, y=y, z=z, ux=uxp, uy=uyp, uz=uzp, w=w, unique_particles=False
    )

    return electrons, dict(x=x, y=y, z=z, ux=uxp, uy=uyp, uz=uzp, w=w)


def _cell_volume(sim):
    """Volume of one cell of the level-0 grid.

    WarpX divides the deposited charge by the product of the cell sizes of the
    dimensions it actually has, so this is the matching volume element in 1D,
    2D and 3D alike.
    """
    return float(np.prod(sim.extension.warpx.Geom(0).data().CellSize()))


@pytest.fixture(scope="function")
def rtol():
    """Callable returning the tolerance for the compiled precision."""
    return _rtol


@pytest.fixture(scope="function")
def uniform_particles():
    """Callable adding a regular lattice of macro particles, see below."""
    return _uniform_particles


@pytest.fixture(scope="function")
def cell_volume():
    """Callable returning the level-0 cell volume of a simulation."""
    return _cell_volume
