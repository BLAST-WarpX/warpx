# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
# License: BSD-3-Clause-LBNL

"""Exercise the tagged splitter directly, independently of AMR tagging."""

import numpy as np
import pytest
from conftest import Config, rtol
from helpers import GRID_CLASS

import pywarpx
from pywarpx import picmi

amr = pywarpx.libwarpx.amr
warpx = pywarpx.libwarpx.libwarpx_so
DIMS = pywarpx.libwarpx.geometry_dim
AXES = {"1d": ("z",), "2d": ("x", "z"), "3d": ("x", "y", "z"), "rz": ("r", "z")}[DIMS]
PPC = {"1d": [5], "2d": [2, 5], "3d": [2, 3, 5], "rz": [2, 7, 5]}[DIMS]
WEIGHTS = np.array([8.0, 16.0, 4.0, 2.0])
VELOCITIES = 1.0e5 * np.array([[1, 2, -3], [4, -2, 4], [-2, 3, 1], [3, -1, 2]])


def make_particles(split_type, case):
    """A zero-density species on several boxes; only four markers are added."""
    ndim = len(AXES)
    bounded = case in ("boundary", "contained_boundary", "roundoff", "pending_boundary")
    field_lo = ["dirichlet" if bounded else "periodic"] * ndim
    field_hi = field_lo.copy()
    particle_lo = ["reflecting" if bounded else "periodic"] * ndim
    particle_hi = particle_lo.copy()
    if DIMS == "rz":
        field_lo[0], field_hi[0] = "none", "dirichlet"
        particle_lo[0], particle_hi[0] = "none", "reflecting"
    grid_class = picmi.CylindricalGrid if DIMS == "rz" else GRID_CLASS[DIMS]
    grid = grid_class(
        number_of_cells=[16] * ndim,
        lower_bound=[0.0] * ndim,
        upper_bound=[1.0] * ndim,
        lower_boundary_conditions=field_lo,
        upper_boundary_conditions=field_hi,
        lower_boundary_conditions_particles=particle_lo,
        upper_boundary_conditions_particles=particle_hi,
        warpx_max_grid_size=8,
        **({"n_azimuthal_modes": 1} if DIMS == "rz" else {}),
    )
    sim = picmi.Simulation(
        solver=picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9),
        max_steps=0,
        verbose=0,
        particle_shape="linear",
        warpx_use_filter=False,
        warpx_amrex_the_arena_is_managed=False,
        warpx_amrex_the_arena_init_size=0,
    )
    species = picmi.Species(
        particle_type="proton",
        name="ions",
        initial_distribution=picmi.UniformDistribution(density=0.0),
    )
    counts = PPC if case == "spacing" else [1] * len(PPC)
    sim.add_species(
        species, layout=picmi.GriddedLayout(n_macroparticle_per_cell=counts)
    )
    sim.initialize_inputs()
    species.species.split_type = split_type
    # A single-level run may enable splitting; parents still need explicit tags.
    species.species.do_splitting = case == "enabled"
    if case == "nongridded":
        species.species.injection_style = "none"
    pywarpx.warpx.do_electrostatic = "none"
    pywarpx.amrex.signal_handling = 0
    pywarpx.amrex.throw_exception = 1
    sim.initialize_warpx()
    return (
        sim,
        sim.particles.get("ions"),
        np.array(counts)[[0, 2] if DIMS == "rz" else slice(None)],
    )


def snapshot(pc):
    """Copy tile arrays to the host, then collect all ranks (including empty ones)."""
    tiles = []
    for pti in warpx.WarpXParIter(pc, 0):
        if pti.num_particles == 0:
            continue
        soa = pti.soa().to_numpy(copy=True)
        tiles.append({**soa.real, **soa.int, "idcpu": soa.idcpu})
    if Config.have_mpi:
        from mpi4py import MPI

        tiles = [tile for rank in MPI.COMM_WORLD.allgather(tiles) for tile in rank]
    return {name: np.concatenate([tile[name] for tile in tiles]) for name in tiles[0]}


def tag_parents(pc, attributes, pending_boundary):
    """Use AMReX's ID packing, including nonzero CPU bits, without host device dereferences."""
    for pti in warpx.WarpXParIter(pc, 0):
        if pti.num_particles == 0:
            continue
        soa = pti.soa()
        host = soa.to_numpy(copy=True)
        selected = host.real["w"] >= 8.0
        ids = amr.unpack_ids(host.idcpu)
        cpus = amr.unpack_cpus(host.idcpu)
        ids[selected] = pc.do_split_particle_id
        cpus[selected] = 7
        ids[host.real["w"] == 2.0] = pc.no_split_particle_id
        amr.pack_ids(host.idcpu, ids)
        amr.pack_cpus(host.idcpu, cpus)
        soa.get_idcpu_data().copy_from_numpy(host.idcpu)
        if attributes:
            soa.get_real_data(pc.get_real_comp_index("split_real")).copy_from_numpy(
                host.real["ux"]
            )
            soa.get_int_data(pc.get_int_comp_index("split_int")).copy_from_numpy(
                host.real["w"].astype(np.int32)
            )
        if pending_boundary:
            positions = host.real[AXES[-1]]
            positions[selected] = 1.001
            soa.get_real_data(pc.get_real_comp_index(AXES[-1])).copy_from_numpy(
                positions
            )


CASES = [("interior", False), ("interior", True)] + [
    (case, True)
    for case in (
        "enabled",
        "spacing",
        "boundary",
        "contained_boundary",
        "roundoff",
        "pending_boundary",
        "nongridded",
        "periodic",
        "angles",
    )
]


@pytest.mark.parametrize("split_type", [0, 1], ids=["diagonal", "axial"])
@pytest.mark.parametrize("case,attributes", CASES)
def test_particle_splitting(split_type, case, attributes):
    if case == "angles" and DIMS != "rz":
        pytest.skip("angular coordinates are specific to RZ")
    sim, pc, counts = make_particles(split_type, case)
    assert sim.extension.warpx.max_level == 0
    if attributes:
        pc.add_real_comp("split_real", True)
        pc.add_int_comp("split_int", True)
    pc.split_particles(0)
    assert pc.size == 0  # Empty containers are valid, too.

    positions = np.tile([0.375, 0.499, 0.625, 0.75], (len(AXES), 1)).T
    if case in ("boundary", "contained_boundary", "roundoff", "periodic"):
        positions[:2] = np.array(
            [0.0625, 0.9375] if case == "contained_boundary" else [0.001, 0.999]
        )[:, None]
        if case == "periodic" and DIMS == "rz":
            positions[:2, 0] = [0.375, 0.625]  # Only z is periodic in RZ.
    if case == "roundoff":
        dtype = np.float32 if Config.precision_particles == "SINGLE" else np.float64
        geom = sim.extension.warpx.Geom(0)
        for i, face in enumerate((0.0, 1.0)):
            value = dtype(face)
            for _ in range(16):
                if geom.insideRoundOffDomain(*([float(value)] * len(AXES))):
                    break
                value = np.nextafter(value, dtype(0.5))
            else:
                pytest.fail(
                    "Could not locate a representable boundary-adjacent particle"
                )
            positions[i] = value
    coords = dict(zip(AXES, positions.T))
    if DIMS == "rz":
        theta = np.array([0.7, -1.2, 0.7, -1.2]) if case == "angles" else np.zeros(4)
        radius = coords.pop("r")
        coords["x"] = radius * np.cos(theta)
        coords["y"] = radius * np.sin(theta)
    # The add_n_particles binding expects contiguous host arrays.
    pc.add_particles(
        **coords,
        ux=VELOCITIES[:, 0].copy(),
        uy=VELOCITIES[:, 1].copy(),
        uz=VELOCITIES[:, 2].copy(),
        w=WEIGHTS,
        unique_particles=False,
    )
    assert pc.size == 4
    tag_parents(pc, attributes, case == "pending_boundary")
    before = snapshot(pc)
    energy_before = pc.sum_particle_energy(local=False)
    pc.split_particles(0)

    if case == "pending_boundary":
        # Splitting must leave parents awaiting physical boundary handling intact.
        after = snapshot(pc)
        for name in before:
            np.testing.assert_array_equal(after[name], before[name])
        pc.apply_boundary_conditions()
        pc.redistribute()
        after = snapshot(pc)
        assert pc.size == 4
        np.testing.assert_allclose(after["w"].sum(), WEIGHTS.sum(), rtol=rtol(), atol=0)
        assert np.all((after[AXES[-1]] >= 0) & (after[AXES[-1]] < 1))
        np.testing.assert_allclose(
            pc.sum_particle_energy(local=False), energy_before, rtol=rtol(), atol=0
        )
        return

    pc.redistribute()
    children = 2 ** len(AXES) if split_type == 0 else 2 * len(AXES)
    assert pc.size == 2 * children + 2
    after = snapshot(pc)
    np.testing.assert_allclose(after["w"].sum(), WEIGHTS.sum(), rtol=rtol(), atol=0)
    np.testing.assert_allclose(
        pc.sum_particle_energy(local=False), energy_before, rtol=rtol(), atol=0
    )
    momentum = np.array([np.dot(after["w"], after[u]) for u in ("ux", "uy", "uz")])
    np.testing.assert_allclose(momentum, WEIGHTS @ VELOCITIES, rtol=rtol(), atol=0)

    tol = rtol()
    for parent in range(4):
        initial = before["ux"] == VELOCITIES[parent, 0]
        selected = after["ux"] == VELOCITIES[parent, 0]
        n = children if parent < 2 else 1
        assert np.count_nonzero(initial) == 1
        assert np.count_nonzero(selected) == n
        for axis, u in enumerate(("ux", "uy", "uz")):
            np.testing.assert_array_equal(
                after[u][selected], np.full(n, VELOCITIES[parent, axis])
            )
        np.testing.assert_allclose(
            after["w"][selected], WEIGHTS[parent] / n, rtol=tol, atol=0
        )
        ids = amr.unpack_ids(after["idcpu"][selected])
        if parent < 2:
            np.testing.assert_array_equal(ids, np.full(n, pc.no_split_particle_id))
        else:
            np.testing.assert_array_equal(
                after["idcpu"][selected], before["idcpu"][initial]
            )
        if attributes:
            np.testing.assert_array_equal(
                after["split_real"][selected], after["ux"][selected]
            )
            np.testing.assert_array_equal(
                after["split_int"][selected], np.full(n, int(WEIGHTS[parent]))
            )
        if DIMS == "rz":
            np.testing.assert_array_equal(
                after["theta"][selected], np.repeat(before["theta"][initial], n)
            )

        delta = np.column_stack([after[a][selected] - before[a][initial] for a in AXES])
        if case == "periodic":
            delta -= np.rint(delta)  # Unwrap the unit periodic domain.
        np.testing.assert_allclose(delta.sum(axis=0), 0, rtol=0, atol=tol * n)
        for a in AXES:
            assert np.all((after[a][selected] >= 0) & (after[a][selected] < 1))
        if parent >= 2:
            np.testing.assert_allclose(delta, 0, rtol=0, atol=tol)
            continue
        offset = 1.0 / (32 * counts)
        moved = np.abs(delta) > tol
        if case in ("boundary", "roundoff"):
            assert np.all(np.abs(delta) <= offset + tol)
        else:
            np.testing.assert_allclose(
                np.abs(delta)[moved],
                np.broadcast_to(offset, delta.shape)[moved],
                rtol=tol,
                atol=tol,
            )
        if case != "roundoff":
            np.testing.assert_array_equal(
                moved.sum(axis=1), np.full(n, len(AXES) if split_type == 0 else 1)
            )

    # No tagged parents remain; a second call must leave all attributes intact.
    pc.split_particles(0)
    pc.redistribute()
    again = snapshot(pc)
    order = np.lexsort(tuple(after.values()))
    order_again = np.lexsort(tuple(again.values()))
    for name in after:
        np.testing.assert_array_equal(again[name][order_again], after[name][order])
