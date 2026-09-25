# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL

import numpy as np
import pytest
from helpers import N_AXES, add_uniform_particles, make_sim

import pywarpx
from pywarpx import picmi

# The built-in real components of a species come first and in a fixed order:
# the positions, then the weight and the three momenta (see PIdx in
# Source/Particles/WarpXParticleContainer.H). Runtime components are appended
# after them, so the first one sits at N_AXES + N_BUILT_IN_REALS.
N_BUILT_IN_REALS = 4  # w, ux, uy, uz

# One previous-position component per axis the geometry has. RZ is absent
# because PhysicalParticleContainer aborts on save_previous_position there.
PREV_POSITION_COMPS = {
    "1d": ("prev_z",),
    "2d": ("prev_x", "prev_z"),
    "3d": ("prev_x", "prev_y", "prev_z"),
}

NEW_PID_VALUE = 5.0

pytestmark = pytest.mark.skipif(
    pywarpx.libwarpx.geometry_dim not in N_AXES,
    reason="make_sim only builds Cartesian geometries",
)


def gather_by_id(species, comp):
    """All local values of the real component ``comp``, ordered by particle id.

    A species is spread over tiles and the order of its particles within a tile
    is not stable across a step, so comparing a per-particle quantity taken at
    two different times has to match the particles up by id. ``to_numpy`` also
    brings the data back from the device on a GPU build.
    """
    comp_index = species.get_real_comp_index(comp)

    values, ids = [], []
    for pti in species.iterator(level=0):
        soa = pti.soa()
        values.append(soa.get_real_data(comp_index).to_numpy(copy=True))
        ids.append(
            pywarpx.libwarpx.amr.unpack_ids(soa.get_idcpu_data().to_numpy(copy=True))
        )

    values = np.concatenate(values)
    return values[np.argsort(np.concatenate(ids))]


@pytest.mark.parametrize("unique_particles", [True, False])
def test_runtime_real_attribute_follows_the_particles(unique_particles):
    """A runtime attribute added from Python survives a time step.

    ``add_real_comp`` appends a component to the species' SoA and
    ``add_particles`` fills it from a keyword argument of the same name. The
    component is registered as communicated, so its values have to follow the
    particles through the push and the redistribution that ends the step.
    """
    sim = make_sim()

    sim.add_species(
        picmi.Species(particle_type="electron", name="electrons"), layout=None
    )

    sim.initialize_inputs()
    sim.initialize_warpx()

    electrons = sim.particles.get("electrons")
    electrons.add_real_comp("newPid")

    # Pin the SoA layout the deposition and push kernels index into: a new
    # runtime component has to land past the built-ins, not among them.
    n_pos = N_AXES[pywarpx.libwarpx.geometry_dim]
    assert electrons.get_real_comp_index("w") == n_pos
    assert electrons.get_real_comp_index("uz") == n_pos + 3
    assert electrons.get_real_comp_index("newPid") == n_pos + N_BUILT_IN_REALS

    n_per_dim = 4
    add_uniform_particles(
        sim,
        "electrons",
        n_per_dim=n_per_dim,
        uz=1.0e7,
        unique_particles=unique_particles,
        newPid=NEW_PID_VALUE,
    )

    # unique_particles decides how the position arrays are read across ranks:
    # True adds them once per rank, False splits them over the ranks. This
    # suite runs on a single rank, where the two agree, so parametrizing it
    # covers both code paths rather than the difference between them.
    n_part = n_per_dim ** N_AXES[pywarpx.libwarpx.geometry_dim]
    assert electrons.size == n_part

    assert np.all(gather_by_id(electrons, "newPid") == NEW_PID_VALUE)

    # the domain is periodic, so the step moves every particle but loses none
    sim.step(1)

    after_step = gather_by_id(electrons, "newPid")
    assert after_step.size == n_part
    assert np.all(after_step == NEW_PID_VALUE)


def test_previous_positions_hold_the_position_before_the_push():
    """``save_previous_position`` stores where each particle was a step ago.

    WarpX registers one ``prev_*`` runtime component per axis of the geometry
    and fills them at the start of the push, before it moves the particle. A
    step therefore leaves them holding exactly the positions the particles had
    when it began, which is what lets a diagnostic reconstruct a trajectory.
    """
    sim = make_sim()

    sim.add_species(
        picmi.Species(
            particle_type="electron",
            name="electrons",
            warpx_save_previous_position=True,
        ),
        layout=None,
    )

    sim.initialize_inputs()
    sim.initialize_warpx()

    electrons = sim.particles.get("electrons")
    dims = pywarpx.libwarpx.geometry_dim
    prev_comps = PREV_POSITION_COMPS[dims]

    # one component per axis, in axis order, right after the built-ins
    n_pos = N_AXES[dims]
    for offset, comp in enumerate(prev_comps):
        assert electrons.get_real_comp_index(comp) == n_pos + N_BUILT_IN_REALS + offset

    add_uniform_particles(sim, "electrons", uz=1.0e7)

    # one step to fill the components, then read the positions they have to
    # reproduce after the next one
    sim.step(1)
    positions = {
        comp: gather_by_id(electrons, comp.removeprefix("prev_")) for comp in prev_comps
    }

    sim.step(1)

    for comp in prev_comps:
        # the push copies the position verbatim, so this is exact rather than
        # approximate, whatever the precision WarpX was compiled with
        np.testing.assert_array_equal(gather_by_id(electrons, comp), positions[comp])

    # guard against the assertion above going vacuous: it only says something
    # as long as the step actually moved the particles along that axis
    moving_comp = prev_comps[-1]  # prev_z, the axis the particles drift along
    assert np.all(
        gather_by_id(electrons, moving_comp.removeprefix("prev_"))
        != positions[moving_comp]
    )
