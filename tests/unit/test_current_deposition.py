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

constants = picmi.constants

# The conservation identities below integrate a density over the grid with a
# Cartesian volume element. RZ deposits with an inverse volume scaling instead,
# so it needs its own test rather than this assertion.
pytestmark = pytest.mark.skipif(
    pywarpx.libwarpx.geometry_dim not in ("1d", "2d", "3d"),
    reason="conservation identity assumes a Cartesian volume element",
)


def _total(mf, geom, dV):
    """Integrate a field component over the (periodic) domain."""
    return mf.sum_unique(comp=0, local=False, period=geom.periodicity()) * dV


@pytest.mark.parametrize("current_deposition_algo", ["direct", "esirkepov"])
def test_current_deposition_conserves_total_current(
    make_sim, uniform_particles, cell_volume, rtol, current_deposition_algo
):
    """Current deposition must conserve the total current of the species.

    Integrating the deposited current density over the grid must return
    ``sum_p q_p w_p v_p``. This holds for the direct deposition, where each
    particle deposits ``q w v`` weighted by shape factors that sum to one, and
    for Esirkepov, which builds J from the displacement ``v dt`` so that its
    grid sum is the same.

    The two algorithms cannot coexist in one initialized simulation, so this
    parametrization also exercises that a second WarpX instance can be created
    in the same Python process (see :meth:`pywarpx.WarpX.finalize`).
    """
    sim = make_sim(current_deposition_algo=current_deposition_algo)

    # relativistic enough that the per-step displacement is a sizeable fraction
    # of a cell: Esirkepov forms differences of shape factors, which loses
    # precision when the displacement is vanishingly small
    uz = 1.0e8
    electrons, p = uniform_particles(sim, uz=uz)

    fields = sim.fields
    for direction in ("x", "y", "z"):
        fields.get("current_fp", direction, 0).set_val(0.0)

    dt = sim.extension.warpx.getdt(0)
    electrons.deposit_current("current_fp", 0, dt, 0.0)

    geom = sim.extension.warpx.Geom(0)
    dV = cell_volume(sim)

    gamma = np.sqrt(1.0 + (p["ux"] ** 2 + p["uy"] ** 2 + p["uz"] ** 2) / constants.c**2)
    expected_jz = -constants.q_e * np.sum(p["w"] * p["uz"] / gamma)

    total_jz = _total(fields.get("current_fp", "z", 0), geom, dV)
    assert np.isclose(total_jz, expected_jz, rtol=rtol())

    # the particles have no transverse momentum, so Jx and Jy must integrate
    # to zero
    for direction in ("x", "y"):
        total = _total(fields.get("current_fp", direction, 0), geom, dV)
        assert np.isclose(total, 0.0, atol=abs(expected_jz) * rtol())


def test_current_deposition_scales_with_weight(
    make_sim, uniform_particles, cell_volume, rtol
):
    """The deposited current must be linear in the macro particle weight."""
    sim = make_sim(current_deposition_algo="direct")

    uz = 1.0e8
    weight = 1.0e6
    electrons, p = uniform_particles(sim, weight=weight, uz=uz)
    # a second set at the same positions, carrying twice the weight
    uniform_particles(sim, weight=2.0 * weight, uz=uz)

    fields = sim.fields
    for direction in ("x", "y", "z"):
        fields.get("current_fp", direction, 0).set_val(0.0)

    dt = sim.extension.warpx.getdt(0)
    electrons.deposit_current("current_fp", 0, dt, 0.0)

    geom = sim.extension.warpx.Geom(0)
    total = _total(fields.get("current_fp", "z", 0), geom, cell_volume(sim))

    gamma = np.sqrt(1.0 + uz**2 / constants.c**2)
    n_part = p["w"].size
    expected = -constants.q_e * n_part * (weight + 2.0 * weight) * uz / gamma

    assert np.isclose(total, expected, rtol=rtol())
