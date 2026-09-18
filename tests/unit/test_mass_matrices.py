# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Remi Lehe
# License: BSD-3-Clause-LBNL

"""Unit tests for the mass matrices of the implicit solvers.

The mass matrices ``S`` are the linear response of the deposited current
density to the electric field, ``dJ = S dE``, which the implicit solvers use
in place of pushing and depositing the particles at every linear iteration.
They are deposited in ``Source/Particles/Deposition/MassMatricesDeposition.H``
and applied in ``ImplicitSolver::ApplyMassMatrices``. The tests below check
them against the result of pushing the particles and depositing the current.
"""

import numpy as np
import pytest
from conftest import rtol
from helpers import N_AXES, add_uniform_particles, make_sim

import pywarpx
from pywarpx import picmi

constants = picmi.constants

# RZ deposits with an inverse volume scaling and rotates the mass matrices
# into cylindrical components; make_sim does not build that geometry yet.
# In 3D, the mass matrix deposition is not yet implemented.
pytestmark = pytest.mark.skipif(
    pywarpx.libwarpx.geometry_dim not in ("1d", "2d"),
    reason="full mass matrices are only implemented in Cartesian 1D and 2D",
)


def _alloc_like(sim, name, template):
    """Register a zeroed vector field with the layout of vector field ``template``.

    Same box arrays, staggering and guard cells for each of the three
    components, so that the new field can stand in for ``template`` wherever
    the C++ side expects that staggering.
    """
    fields = sim.fields
    for direction in ("x", "y", "z"):
        mf = fields.get(template, direction, 0)
        fields.alloc_init(
            name,
            direction,
            0,
            mf.box_array(),
            mf.dm(),
            mf.n_comp,
            mf.n_grow_vect,
            0.0,
            redistribute=False,
            redistribute_on_remake=False,
        )


def _fill_periodic_random(mf, n_cell, rng, amplitude):
    """Fill ``mf``, guard cells included, with a random periodic field.

    One random value is drawn per cell of the domain and every point of the
    MultiFab, whether valid or guard, nodal or cell-centered, reads the value
    of the cell it wraps to. A nodal point on the upper boundary therefore
    equals its periodic image on the lower boundary, and the guard cells hold
    what a periodic fill would put there, without any communication.
    """
    table = rng.uniform(-amplitude, amplitude, size=n_cell)
    # imesh puts cell-centered points at half-integers; floor gives the cell
    wrapped = [
        np.floor(mf.imesh(idir, include_ghosts=True)).astype(int) % n_cell[idir]
        for idir in range(len(n_cell))
    ]
    mf[()] = table[np.ix_(*wrapped)]


@pytest.mark.parametrize("particle_shape", ["linear", "quadratic", "cubic"])
def test_mass_matrices_match_push_and_deposit(particle_shape):
    """``S dE`` must equal the current deposited after a push in ``dE``.

    The particles start at rest (``u^n=0``), and ``dE`` is chosen low enough
    that they remain non-relativistic over one timestep. This so that ``u^{n+1}``
    from the Boris pusher remains linear in ``dE`` (relativistic effects introduce
    non-linearities in ``dE``) but also because WarpX's implementation of
    the mass matrix does not yet fully take into account all relativistic effects.

    The mass matrices give the response of the time-centered current
    ``(u^n + u^{n+1}) / 2``, whereas the push from rest leaves ``u^{n+1}`` on
    the particles, hence the factor 1/2 on the reference.
    """
    n_axes = N_AXES[pywarpx.libwarpx.geometry_dim]
    # 8 cells and 4 cells per box (two boxes per axis), so that contributions
    # crossing a box boundary and the periodic boundary are both exercised.
    # The direct deposition also makes WarpX gather with plain shape factors
    # (no Galerkin correction), which is what the mass matrices assume.
    n_cell = [8] * n_axes
    sim = make_sim(
        n_cell=n_cell,
        max_grid_size=4,
        particle_shape=particle_shape,
        current_deposition_algo="direct",
    )

    # the mass matrices are only allocated by an evolve scheme that uses them
    sim.evolve_scheme = picmi.ThetaImplicitEMEvolveScheme(
        nonlinear_solver=picmi.NewtonNonlinearSolver(
            linear_solver=picmi.GMRESLinearSolver(),
            use_mass_matrices_jacobian=True,
        ),
        theta=0.5,
    )

    sim.add_species(
        picmi.Species(particle_type="electron", name="electrons"), layout=None
    )
    sim.initialize_inputs()
    sim.initialize_warpx()

    warpx = sim.extension.warpx
    fields = sim.fields
    dt = warpx.getdt(0)

    # the particles start at rest, see the docstring
    add_uniform_particles(sim, "electrons")
    electrons = sim.particles.get("electrons")

    # A uniform magnetic field with all three components, strong enough that
    # the normalized gyration ``b = q dt B / (2 m)`` is of order one:
    # This ensures that the terms associated with the magnetic field in the
    # mass matrix have a significant impact in this test.
    b_unit = 2.0 * constants.m_e / (constants.q_e * dt)
    for direction, b in zip(("x", "y", "z"), (0.6, -0.8, 1.1)):
        fields.get("Bfield_fp", direction, 0).set_val(b * b_unit)

    # Initialize the saved momentum at time n (read by the implicit deposition routine)
    warpx.save_particles_at_implicit_step_start()

    solver = warpx.implicit_solver()
    warpx.deposit_mass_matrices()
    warpx.sync_mass_matrices()  # Sum the guard cells of the mass matrices into the valid cells
    solver.finish_mass_matrices()  # Fill the second half of the diagonal mass matrices by symmetry

    # ``dE`` gets the layout of ``Efield_fp``, guard cells included: the band of
    # each (J, E) pair reaches nox + 1 cells, so this also checks that WarpX gives
    # E one more guard cell than J, without which ApplyMassMatrices would silently
    # truncate the band and drop its outermost terms.
    _alloc_like(sim, "dE", "Efield_fp")
    _alloc_like(sim, "dJ", "current_fp")

    # The amplitude keeps the push non-relativistic: q dE dt / m is a fraction
    # of a meter per second
    rng = np.random.default_rng(seed=42)
    for direction in ("x", "y", "z"):
        _fill_periodic_random(fields.get("dE", direction, 0), n_cell, rng, 1.0)

    # mass matrices: dJ = S dE
    solver.apply_mass_matrices(
        fields.mr_levels_alldirs("dJ", 0),
        fields.mr_levels_alldirs("dE", 0),
        zero_out_first=True,
    )

    # reference: push from rest in (dE, B), then deposit the current into the
    # (so far unused) current_fp
    sim.particles.push_p(
        0,
        dt,
        *(fields.get("dE", direction, 0) for direction in ("x", "y", "z")),
        *(fields.get("Bfield_fp", direction, 0) for direction in ("x", "y", "z")),
    )
    for direction in ("x", "y", "z"):
        fields.get("current_fp", direction, 0).set_val(0.0)
    electrons.deposit_current("current_fp", 0, dt, 0.0)

    for direction in ("x", "y", "z"):
        # ``dJ`` contains the time-centered current ``(u^n + u^{n+1}) / 2`` (with u^n = 0 here)
        # whereas ``current_fp`` contains the current for `u^{n+1}`` (result of push+deposit)
        # Hence the multiplication by 0.5 when comparing ``dJ`` and ``current_fp``.
        dj_mass_matrices = fields.get("dJ", direction, 0)[...]
        dj_reference = 0.5 * fields.get("current_fp", direction, 0)[...]

        scale = np.max(np.abs(dj_reference))
        assert scale > 0.0
        error = np.max(np.abs(dj_mass_matrices - dj_reference)) / scale
        assert error <= rtol(), (
            f"J{direction}: max |dJ_mm - dJ_ref| / max |dJ_ref| = {error}"
        )
