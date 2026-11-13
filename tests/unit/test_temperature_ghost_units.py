# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
# License: BSD-3-Clause-LBNL

"""Temperature conversion must include the physical guards read by filtering."""

from itertools import product
from math import comb

import numpy as np
import pytest
from conftest import rtol

import pywarpx
from pywarpx import picmi

DIMS = pywarpx.libwarpx.geometry_dim
pytestmark = pytest.mark.skipif(DIMS not in ("rz", "3d"), reason="RZ and 3D coverage")
N_CELL = 16
MASS = 2.0e-27
KB = 1.380649e-23


def _make_sim(periodic, passes):
    ndim = 2 if DIMS == "rz" else 3
    field_lo = ["periodic" if periodic else "dirichlet"] * ndim
    particle_lo = ["periodic" if periodic else "reflecting"] * ndim
    field_hi, particle_hi = field_lo.copy(), particle_lo.copy()
    kwargs = {}
    grid_type = picmi.Cartesian3DGrid
    if DIMS == "rz":
        grid_type = picmi.CylindricalGrid
        kwargs["n_azimuthal_modes"] = 1
        field_lo[0] = particle_lo[0] = "none"
    grid = grid_type(
        number_of_cells=[N_CELL] * ndim,
        lower_bound=[0.0] * ndim,
        upper_bound=[float(N_CELL)] * ndim,
        lower_boundary_conditions=field_lo,
        upper_boundary_conditions=field_hi,
        lower_boundary_conditions_particles=particle_lo,
        upper_boundary_conditions_particles=particle_hi,
        warpx_max_grid_size=8,
        **kwargs,
    )
    solver = picmi.HybridPICSolver(
        grid=grid,
        Te=0.0,
        n0=1.0,
        gamma=1.0,
        n_floor=1.0,
        plasma_resistivity="0.0",
        plasma_hyper_resistivity="0.0",
        solve_electron_energy_equation=False,
        include_joule_heating=False,
        electron_ion_relaxation_rate="0.0",
        substeps=1,
    )
    sim = picmi.Simulation(
        solver=solver,
        time_step_size=1.0e-15,
        max_steps=1,
        verbose=0,
        particle_shape="cubic",
        warpx_use_filter=passes is not None,
        warpx_current_deposition_algo="direct",
        warpx_used_inputs_file="used_inputs",
    )
    sim.add_species(
        picmi.Species(
            name="ions",
            mass=MASS,
            charge=picmi.constants.q_e,
            warpx_do_not_push=True,
            warpx_do_not_gather=True,
            warpx_do_temperature_deposition=True,
        ),
        layout=None,
    )
    sim.initialize_inputs()
    if passes is not None:
        pywarpx.warpx.filter_npass_each_dir = passes
    pywarpx.warpx.do_electrostatic = "none"
    pywarpx.amrex.the_arena_init_size = 0
    pywarpx.amrex.throw_exception = 1
    pywarpx.amrex.signal_handling = 0
    sim.write_input_file("inputs")
    sim.initialize_warpx()
    return sim


def _pair_moments(positions, lo, hi, nodal, periodic):
    """Independent centered cubic B-splines for two samples at each position."""
    weight = np.zeros(tuple(hi - lo + 1))
    count = np.zeros_like(weight, dtype=int)
    grid = np.meshgrid(
        *(np.arange(a, b + 1) + 0.5 * (1 - n) for a, b, n in zip(lo, hi, nodal)),
        indexing="ij",
    )
    for position in positions:
        shape = np.ones_like(weight)
        for x, center in zip(grid, position):
            distance = x - center
            if periodic:
                # The support radius is less than half the periodic length,
                # so exactly one image can contribute at any grid point.
                distance = (distance + N_CELL / 2) % N_CELL - N_CELL / 2
            distance = np.abs(distance)
            shape *= np.where(
                distance < 1,
                (4 - 6 * distance**2 + 3 * distance**3) / 6,
                np.maximum(2 - distance, 0) ** 3 / 6,
            )
        weight += 2 * shape
        count += 2 * (shape > 0)
    # Distinct pairs must not overlap, including their periodic images.
    assert set(np.unique(count)).issubset({0, 2})
    return weight, count


def _binomial_filter(value, passes):
    """Apply the analytic tensor-product stencil, zero outside this FAB."""
    for axis, count in enumerate(passes):
        if count == 0:
            continue
        padding = [(0, 0)] * value.ndim
        padding[axis] = (count, count)
        padded = np.pad(value, padding)
        result = np.zeros_like(value)
        for offset in range(2 * count + 1):
            indices = [slice(None)] * value.ndim
            indices[axis] = slice(offset, offset + value.shape[axis])
            result += comb(2 * count, offset) / 4**count * padded[tuple(indices)]
        value = result
    return value


def _expected_temperature(positions, lo, hi, nodal, periodic, passes):
    _, count = _pair_moments(positions, lo, hi, nodal, periodic)
    value = (count > 0).astype(float)
    if passes is None:
        return value
    value = _binomial_filter(value, passes)

    # The native post-filter FillBoundary replaces internal/periodic ghosts
    # with values from valid donor nodes, whose stencil has complete support.
    radius = np.array(passes)
    _, expanded = _pair_moments(positions, lo - radius, hi + radius, nodal, periodic)
    complete = _binomial_filter((expanded > 0).astype(float), passes)
    complete = complete[tuple(slice(p, p + n) for p, n in zip(passes, value.shape))]
    donor = np.ones_like(value, dtype=bool)
    if not periodic:
        indices = np.meshgrid(
            *(np.arange(a, b + 1) for a, b in zip(lo, hi)), indexing="ij"
        )
        for x, n in zip(indices, nodal):
            donor &= (x >= 0) & (x < N_CELL + n)
    return np.where(donor, complete, value)


def _host_array(field, mfi):
    # Explicit host copy keeps the test usable on GPU builds as well.
    value = field.array(mfi).to_numpy(copy=True)
    ndim = 2 if DIMS == "rz" else 3
    value = value[(slice(None),) * ndim + (0,) * (4 - ndim)]
    return value


@pytest.mark.parametrize("case", ["unfiltered", "one_pass", "anisotropic", "periodic"])
def test_temperature_ghost_units(case):
    if case == "periodic" and DIMS == "rz":
        pytest.skip("The RZ radial boundary is an axis, not a periodic seam")
    ndim = 2 if DIMS == "rz" else 3
    periodic = case == "periodic"
    passes = None if case == "unfiltered" else [1] * ndim
    if case in ("anisotropic", "periodic"):
        passes = list(range(1, ndim + 1))
    sim = _make_sim(periodic, passes)

    # Identical positive weights and positions remove any estimator dependency:
    # both raw-count and effective-weight corrections give exactly 2*v**2.
    coordinates = (0.25, 8.25) if periodic else (0.25, 8.25, 15.75)
    positions = np.array(list(product(coordinates, repeat=ndim)))
    pair_positions = np.repeat(positions, 2, axis=0)
    count = len(pair_positions)
    signs = np.tile([1.0, -1.0], len(positions))
    sim.particles.get("ions").add_particles(
        x=pair_positions[:, 0].copy(),
        y=np.zeros(count) if DIMS == "rz" else pair_positions[:, 1].copy(),
        z=pair_positions[:, -1].copy(),
        ux=signs,
        uy=2 * signs,
        uz=3 * signs,
        w=np.ones(count),
        unique_particles=False,
    )
    assert sim.particles.get("ions").size == count
    sim.step(1)
    assert sim.particles.get("ions").size == count

    directions = ("r", "theta", "z") if DIMS == "rz" else ("x", "y", "z")
    for component, direction in enumerate(directions):
        temperature = sim.fields.get("T_ions", direction, 0)
        guards = np.array(tuple(temperature.n_grow_vect))
        expected_guards = 3 + np.array(passes if passes is not None else [0] * ndim)
        np.testing.assert_array_equal(guards, expected_guards)
        nodal = np.array([int(temperature.is_nodal(i)) for i in range(ndim)])
        # Inputs are proper velocities; temperature uses ordinary velocity.
        velocity_squared = (component + 1) ** 2 / (1 + 14 / picmi.constants.c**2)
        scale = 2 * velocity_squared * MASS / KB
        # Exhaust native iterators before asserting: pytest keeps a failed
        # assertion's traceback across the next WarpX initialization.
        weight_field = sim.fields.get("variance_buffer_w_ions", direction, 0)
        residual_field = sim.fields.get("variance_buffer_w2_ions", direction, 0)
        snapshots = []
        for mfi in temperature:
            box = mfi.fabbox()
            snapshots.append(
                (
                    np.array(tuple(box.small_end)),
                    np.array(tuple(box.big_end)),
                    _host_array(temperature, mfi),
                    _host_array(weight_field, mfi),
                    _host_array(residual_field, mfi),
                )
            )
        for lo, hi, actual, actual_weight, actual_residual in snapshots:
            assert np.isfinite(actual).all()
            assert np.isfinite(actual_weight).all()
            assert np.isfinite(actual_residual).all()
            weight, _ = _pair_moments(positions, lo, hi, nodal, periodic)
            # The unfiltered moment buffers independently verify both members
            # reach the same cubic support at physical, box and periodic seams.
            np.testing.assert_allclose(actual_weight, weight, rtol=rtol(), atol=rtol())
            np.testing.assert_allclose(
                actual_residual,
                weight * velocity_squared,
                rtol=rtol(),
                atol=rtol() * velocity_squared,
            )
            expected = (
                _expected_temperature(positions, lo, hi, nodal, periodic, passes)
                * scale
            )
            np.testing.assert_allclose(
                actual, expected, rtol=rtol(), atol=rtol() * scale
            )
