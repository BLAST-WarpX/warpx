# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL

import numpy as np
import pytest

import pywarpx
from pywarpx import picmi

# tolerances for the conservation identities asserted by the unit tests
RTOL = {"SINGLE": 1.0e-5, "DOUBLE": 1.0e-12}


def rtol():
    """Relative tolerance matching the precision WarpX was compiled with."""
    return RTOL[pywarpx.libwarpx.Config.precision]


@pytest.fixture(scope="function")
def make_sim():
    """Factory for a minimal, initialized 3D WarpX simulation.

    Returns a callable so that a test can create a simulation with the exact
    numerics it wants to exercise. Only one simulation may live at a time; the
    ``warpx_lifecycle`` fixture in the parent ``conftest.py`` tears it down
    again after each test, which is what makes parametrizing over e.g. the
    current deposition algorithm possible.
    """

    def _make(
        n_cell=(16, 16, 16),
        lower_bound=(-1.0e-3, -1.0e-3, -1.0e-3),
        upper_bound=(1.0e-3, 1.0e-3, 1.0e-3),
        max_grid_size=8,
        particle_shape="quadratic",
        current_deposition_algo=None,
        dt=None,
    ):
        # dt=None lets WarpX pick the CFL-limited time step, which keeps the
        # per-step particle displacement a sizeable fraction of a cell
        grid = picmi.Cartesian3DGrid(
            number_of_cells=list(n_cell),
            lower_bound=list(lower_bound),
            upper_bound=list(upper_bound),
            lower_boundary_conditions=["periodic"] * 3,
            upper_boundary_conditions=["periodic"] * 3,
            lower_boundary_conditions_particles=["periodic"] * 3,
            upper_boundary_conditions_particles=["periodic"] * 3,
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
            layout=picmi.GriddedLayout(n_macroparticle_per_cell=[0, 0, 0], grid=grid),
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


def uniform_particles(sim, n_per_dim=4, weight=1.0e6, ux=0.0, uy=0.0, uz=0.0):
    """Add a regular lattice of macro particles to the ``electrons`` species.

    The particles are placed strictly inside the domain, offset from both cell
    centers and nodes, so that the shape factors of every deposition order
    spread charge over more than one cell.

    Returns the particle container together with the position and momentum
    arrays that were used, as they are needed to form the expected values.
    """
    geom = sim.extension.warpx.Geom(0)
    lo = np.array(geom.ProbLo())
    hi = np.array(geom.ProbHi())

    # fractional positions in (0, 1), avoiding the domain boundaries
    frac = (np.arange(n_per_dim) + 0.37) / n_per_dim
    fx, fy, fz = np.meshgrid(frac, frac, frac, indexing="ij")

    x = (lo[0] + fx * (hi[0] - lo[0])).ravel()
    y = (lo[1] + fy * (hi[1] - lo[1])).ravel()
    z = (lo[2] + fz * (hi[2] - lo[2])).ravel()

    n_part = x.size
    w = np.full(n_part, weight)
    uxp = np.full(n_part, ux)
    uyp = np.full(n_part, uy)
    uzp = np.full(n_part, uz)

    electrons = sim.particles.get("electrons")
    electrons.add_particles(
        x=x, y=y, z=z, ux=uxp, uy=uyp, uz=uzp, w=w, unique_particles=False
    )

    return electrons, dict(x=x, y=y, z=z, ux=uxp, uy=uyp, uz=uzp, w=w)


def cell_volume(sim):
    """Volume of one cell of the level-0 grid."""
    return float(np.prod(sim.extension.warpx.Geom(0).data().CellSize()))
