# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL

import numpy as np
import pytest
from conftest import rtol
from helpers import N_AXES, add_uniform_particles, make_sim

import pywarpx
from pywarpx import picmi

constants = picmi.constants

# The conservation identities below integrate a density over the grid with a
# Cartesian volume element. RZ deposits with an inverse volume scaling instead,
# so it needs its own test rather than this assertion.
pytestmark = pytest.mark.skipif(
    pywarpx.libwarpx.geometry_dim not in ("1d", "2d", "3d"),
    reason="conservation identity assumes a Cartesian volume element",
)


@pytest.mark.parametrize("current_deposition_algo", ["direct", "esirkepov"])
def test_current_deposition_conserves_total_current(current_deposition_algo):
    """Current deposition must conserve the total current of the species.

    Integrating the deposited current density over the grid must return
    ``sum_p q_p w_p v_p``. This holds for the direct deposition, where each
    particle deposits ``q w v`` weighted by shape factors that sum to one, and
    for Esirkepov, which builds J from the displacement ``v dt`` so that its
    grid sum is the same.
    """
    sim = make_sim(current_deposition_algo=current_deposition_algo)

    sim.add_species(
        picmi.Species(particle_type="electron", name="electrons"), layout=None
    )

    sim.initialize_inputs()
    sim.initialize_warpx()

    uz = 1.0e8
    n_per_dim = 4
    weight = 1.0e6
    add_uniform_particles(sim, "electrons", n_per_dim=n_per_dim, weight=weight, uz=uz)

    n_part = n_per_dim ** N_AXES[pywarpx.libwarpx.geometry_dim]
    electrons = sim.particles.get("electrons")
    fields = sim.fields
    for direction in ("x", "y", "z"):
        fields.get("current_fp", direction, 0).set_val(0.0)

    dt = sim.extension.warpx.getdt(0)
    electrons.deposit_current("current_fp", 0, dt, 0.0)

    gamma = np.sqrt(1.0 + uz**2 / constants.c**2)
    expected_jz = -constants.q_e * n_part * weight * uz / gamma

    # a current component is integrated over the domain by summing its unique nodes,
    # times the cell volume. Passing the periodicity matters, as J is nodal in at
    # least one direction and the boundary nodes would otherwise be counted twice
    geom = sim.extension.warpx.Geom(0)
    cell_volume = float(np.prod(geom.data().CellSize()))

    jz_multifab = fields.get("current_fp", "z", 0)
    total_jz = (
        jz_multifab.sum_unique(comp=0, local=False, period=geom.periodicity())
        * cell_volume
    )
    assert np.isclose(total_jz, expected_jz, rtol=rtol(), atol=0.0)

    # the particles have no transverse momentum, so Jx and Jy must integrate
    # to zero. This one compares against zero, so it needs an absolute
    # tolerance, scaled to the current that is actually flowing.
    for direction in ("x", "y"):
        j_multifab = fields.get("current_fp", direction, 0)
        total_j = (
            j_multifab.sum_unique(comp=0, local=False, period=geom.periodicity())
            * cell_volume
        )
        assert np.isclose(total_j, 0.0, atol=abs(expected_jz) * rtol())


def test_current_deposition_sums_mixed_weights():
    """Particles of differing weight must all contribute to the current.

    The parametrized test above only ever deposits a single weight, so this
    covers the weighting itself: two sets of particles at the same positions,
    carrying ``w`` and ``2 w``, have to sum to the current of ``3 w``.
    """
    sim = make_sim(current_deposition_algo="direct")

    sim.add_species(
        picmi.Species(particle_type="electron", name="electrons"), layout=None
    )

    sim.initialize_inputs()
    sim.initialize_warpx()

    uz = 1.0e8
    n_per_dim = 4
    weight = 1.0e6
    add_uniform_particles(sim, "electrons", n_per_dim=n_per_dim, weight=weight, uz=uz)
    add_uniform_particles(
        sim, "electrons", n_per_dim=n_per_dim, weight=2.0 * weight, uz=uz
    )

    n_part = n_per_dim ** N_AXES[pywarpx.libwarpx.geometry_dim]
    electrons = sim.particles.get("electrons")
    fields = sim.fields
    for direction in ("x", "y", "z"):
        fields.get("current_fp", direction, 0).set_val(0.0)

    dt = sim.extension.warpx.getdt(0)
    electrons.deposit_current("current_fp", 0, dt, 0.0)

    gamma = np.sqrt(1.0 + uz**2 / constants.c**2)
    expected = -constants.q_e * n_part * (weight + 2.0 * weight) * uz / gamma

    # integrate Jz over the domain: sum its unique nodes (the periodicity keeps the
    # nodes on the periodic boundary from being counted twice), times the cell volume
    geom = sim.extension.warpx.Geom(0)
    cell_volume = float(np.prod(geom.data().CellSize()))
    jz_multifab = fields.get("current_fp", "z", 0)
    total_jz = (
        jz_multifab.sum_unique(comp=0, local=False, period=geom.periodicity())
        * cell_volume
    )

    assert np.isclose(total_jz, expected, rtol=rtol(), atol=0.0)
