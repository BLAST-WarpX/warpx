# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL

import numpy as np
import pytest

from pywarpx import picmi

constants = picmi.constants


@pytest.mark.parametrize("particle_shape", ["linear", "quadratic", "cubic"])
def test_charge_deposition_conserves_total_charge(
    make_sim, uniform_particles, cell_volume, rtol, particle_shape
):
    """Charge deposition must conserve the total charge of the species.

    Whatever the B-spline order, the shape factors of a macro particle sum to
    one, so summing the deposited charge density over the grid must return the
    total charge carried by the particles. Each order is a separately compiled
    kernel in ``Source/Particles/Deposition/ChargeDeposition.H``.
    """
    sim = make_sim(particle_shape=particle_shape)

    n_per_dim = 4
    weight = 1.0e6
    n_part = n_per_dim**3
    electrons, _ = uniform_particles(sim, n_per_dim=n_per_dim, weight=weight)

    assert electrons.size == n_part

    # cross-check WarpX's own particle-side sum against the analytic value, so
    # that a failure below can be attributed to the deposition and not to the
    # particles that were added
    expected_charge = -constants.q_e * n_part * weight
    assert np.isclose(
        electrons.sum_particle_charge(local=False), expected_charge, rtol=rtol()
    )

    # allocates a nodal MultiFab, resets it, deposits, sums the guard cells and
    # applies the boundary/volume treatment
    rho = electrons.get_charge_density(lev=0, local=False)

    # rho is nodal and the domain is periodic: without passing the periodicity,
    # the nodes shared across the periodic boundary would be counted twice
    geom = sim.extension.warpx.Geom(0)
    deposited_charge = rho.sum_unique(
        comp=0, local=False, period=geom.periodicity()
    ) * cell_volume(sim)

    assert np.isclose(deposited_charge, expected_charge, rtol=rtol())


def test_charge_deposition_is_negative_for_electrons(make_sim, uniform_particles):
    """Electrons must deposit a negative charge density everywhere."""
    sim = make_sim()
    electrons, _ = uniform_particles(sim)

    rho = electrons.get_charge_density(lev=0, local=False)

    assert rho.max(0) <= 0.0
    assert rho.min(0) < 0.0
