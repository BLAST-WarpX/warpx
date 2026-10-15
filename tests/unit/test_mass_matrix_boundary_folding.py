# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
# License: BSD-3-Clause-LBNL

"""Prepare the reference current for a PC mass-matrix boundary-folding test.

For now this exercises only the unfolded path: sum synthetic per-box
coefficients, extend random E through its BCs, compute J0 = sigma_pc E,
and apply the current BCs. Folding sigma_pc and comparing J1 with J0
will be added separately.
"""

import numpy as np
import pytest
from conftest import rtol
from helpers import make_sim

import pywarpx
from pywarpx import picmi

pytestmark = pytest.mark.skipif(
    pywarpx.libwarpx.geometry_dim != "2d", reason="this setup uses Cartesian XZ"
)


def _array_module():
    config = pywarpx.libwarpx.libwarpx_so.Config
    if not config.have_gpu:
        return np
    if config.gpu_backend == "SYCL":
        import dpnp

        return dpnp
    import cupy

    return cupy


def _nonzero_random(rng, shape):
    """Signed values of order one, with every magnitude bounded away from zero."""
    return rng.choice([-1.0, 1.0], size=shape) * rng.uniform(0.5, 1.5, size=shape)


def _fill_quadratic_pc(mf, direction, xp):
    """Randomize every coefficient supported by particles inside each box.

    A quadratic shape covers three consecutive indices. For a particle
    inside a box's cells, its leftmost shape index ranges from cell_lo - 1
    to cell_hi (nodal) or cell_hi - 1 (centered). A row and its E column
    are supported precisely when some such shape window contains both.
    This leaves unsupported ghost coefficients zero without depositing particles.
    """
    order = 2
    n_stencil = 2 * order + 1
    assert mf.n_comp == n_stencil**2
    for mfi in mf:
        valid = mfi.validbox()
        fab = mfi.fabbox()
        values = mf.array(mfi).to_xp()
        values[...] = 0.0
        rows = [
            np.arange(fab.small_end[axis], fab.big_end[axis] + 1)
            for axis in (0, 1)
        ]
        left_lo = [valid.small_end[axis] - 1 for axis in (0, 1)]
        # valid_hi is cell_hi + 1 for nodal data, and cell_hi for centered data.
        left_hi = [valid.big_end[axis] - 1 for axis in (0, 1)]
        rng = np.random.default_rng(
            [42, direction, valid.small_end[0], valid.small_end[1]]
        )
        for dz in range(-order, order + 1):
            for dx in range(-order, order + 1):
                supported = []
                for axis, offset in enumerate((dx, dz)):
                    column = rows[axis] + offset
                    supported.append(
                        (np.minimum(rows[axis], column) >= left_lo[axis])
                        & (np.maximum(rows[axis], column) - order <= left_hi[axis])
                    )
                mask = supported[0][:, None] & supported[1][None, :]
                coefficient = dx + order + n_stencil * (dz + order)
                random = _nonzero_random(rng, mask.shape)
                values[:, :, 0, coefficient] = xp.asarray(np.where(mask, random, 0.0))


def _fill_random_e(mf, rng):
    """Fill valid E with nonzero random values; communication supplies guards."""
    mf.set_val(0.0)
    shape = tuple(len(mf.imesh(axis)) for axis in (0, 1))
    mf[...] = _nonzero_random(rng, shape)


def _apply_unfolded_pc(sigma, efield, current, xp):
    """Direct diagonal-block stencil sum on valid and guard current rows.

    Array4 views retain the unused third spatial axis, so the XZ plane is
    [:, :, 0, component]. Component offsets flatten with X varying fastest.
    Reads are restricted to the E FAB, as in ApplyMassMatrices. Outer internal
    J guards need not have a complete stencil. Check that no nonzero term is
    clipped on valid rows or physical X-ghost rows needed for the reference.
    """
    order = 2
    n_stencil = 2 * order + 1
    current.set_val(0.0)
    domain = current.box_array().minimal_box()
    for mfi in current:
        valid = mfi.validbox()
        s = sigma.array(mfi).to_xp()
        e = efield.array(mfi).to_xp()[:, :, 0, 0]
        j = current.array(mfi).to_xp()[:, :, 0, 0]
        guard_shift = [
            efield.n_grow_vect[axis] - current.n_grow_vect[axis] for axis in (0, 1)
        ]
        ng_x, ng_z = (current.n_grow_vect[axis] for axis in (0, 1))
        xlo = 0 if valid.small_end[0] == domain.small_end[0] else ng_x
        xhi = (
            j.shape[0]
            if valid.big_end[0] == domain.big_end[0]
            else j.shape[0] - ng_x
        )
        required = np.zeros(j.shape, dtype=bool)
        required[xlo:xhi, ng_z : j.shape[1] - ng_z] = True
        for dz in range(-order, order + 1):
            for dx in range(-order, order + 1):
                shift = (dx + guard_shift[0], dz + guard_shift[1])
                lo = [max(0, -shift[axis]) for axis in (0, 1)]
                hi = [
                    min(j.shape[axis], e.shape[axis] - shift[axis]) for axis in (0, 1)
                ]
                row = tuple(slice(lo[axis], hi[axis]) for axis in (0, 1))
                column = tuple(
                    slice(lo[axis] + shift[axis], hi[axis] + shift[axis])
                    for axis in (0, 1)
                )
                coefficient = dx + order + n_stencil * (dz + order)
                uncovered = required.copy()
                uncovered[row] = False
                assert not bool(
                    xp.any(s[:, :, 0, coefficient][xp.asarray(uncovered)] != 0.0)
                ), f"J0 stencil at offset {(dx, dz)} exceeds the available E guards"
                j[row] += s[:, :, 0, coefficient][row] * e[column]


def _pec_reference(mf, raw_current, n_cell_x, sign):
    """Expected valid current after adding signed physical X-guard contributions."""
    ng_x, ng_z = (mf.n_grow_vect[axis] for axis in (0, 1))
    nx, nz = (len(mf.imesh(axis)) for axis in (0, 1))
    nodal_x = int(mf.box_array().ix_type().node_centered(0))
    expected = raw_current[ng_x : ng_x + nx, ng_z : ng_z + nz].copy()
    for ix in range(nx):
        for boundary in (0, n_cell_x):
            mirror = 2 * boundary - (1 - nodal_x) - ix
            # Only physical guards and a nodal boundary row fold into this row.
            if 0 <= mirror < nx and mirror != ix:
                continue
            if -ng_x <= mirror < nx + ng_x:
                expected[ix, :] += sign * raw_current[mirror + ng_x, ng_z : ng_z + nz]
    return expected


def test_unfolded_pc_current_pec_x_periodic_z():
    """Construct J0 with PEC on both X faces and periodic Z, without folding PC."""
    n_cell = [16, 16]
    sim = make_sim(
        n_cell=n_cell,
        max_grid_size=8,
        particle_shape="quadratic",
        current_deposition_algo="direct",
    )
    grid = sim.solver.grid
    grid.lower_boundary_conditions = ["pec", "periodic"]
    grid.upper_boundary_conditions = ["pec", "periodic"]
    grid.lower_boundary_conditions_particles = ["absorbing", "periodic"]
    grid.upper_boundary_conditions_particles = ["absorbing", "periodic"]
    sim.evolve_scheme = picmi.ThetaImplicitEMEvolveScheme(
        nonlinear_solver=picmi.NewtonNonlinearSolver(
            linear_solver=picmi.GMRESLinearSolver(),
            pc_type=picmi.JacobiPreconditioner(),
            use_mass_matrices_jacobian=True,
            use_mass_matrices_pc=True,
            mass_matrices_pc_width=2,
        ),
        theta=0.5,
    )
    # An empty species enables particle gather/deposition guard-cell allocation.
    sim.add_species(picmi.Species(particle_type="electron", name="electrons"), layout=None)
    sim.initialize_inputs()
    sim.initialize_warpx()

    warpx = sim.extension.warpx
    fields = sim.fields
    xp = _array_module()
    for idir, direction in enumerate(("x", "y", "z")):
        sigma = fields.get("MassMatrices_PC", direction, 0)
        _fill_quadratic_pc(sigma, idir, xp)
        template = fields.get("current_fp", direction, 0)
        fields.alloc_init(
            "J0",
            direction,
            0,
            template.box_array(),
            template.dm(),
            template.n_comp,
            template.n_grow_vect,
            0.0,
            redistribute=False,
            redistribute_on_remake=False,
        )

    # Sum box-local deposition contributions once, leaving physical BCs unfolded.
    warpx.sync_mass_matrices_pc()
    rng = np.random.default_rng(43)
    for direction in ("x", "y", "z"):
        _fill_random_e(fields.get("Efield_fp", direction, 0), rng)
    warpx.fill_boundary_e()
    for direction in ("y", "z"):
        e = fields.get("Efield_fp", direction, 0)[...]
        assert np.all(e[[0, -1], :] == 0.0), f"E{direction}: PEC Dirichlet nodes"

    expected = {}
    for direction in ("x", "y", "z"):
        j0 = fields.get("J0", direction, 0)
        _apply_unfolded_pc(
            fields.get("MassMatrices_PC", direction, 0),
            fields.get("Efield_fp", direction, 0),
            j0,
            xp,
        )
        expected[direction] = _pec_reference(
            j0, j0[()], n_cell[0], sign=1.0 if direction == "x" else -1.0
        )

    # Sigma was already summed. Apply only physical current BCs to J0 here.
    warpx.apply_jfield_boundary("J0")
    for direction in ("x", "y", "z"):
        j0 = fields.get("J0", direction, 0)[...]
        assert np.all(np.isfinite(j0)), f"J0{direction}: non-finite reference current"
        scale = np.max(np.abs(expected[direction]))
        assert scale > 0.0, f"J0{direction}: reference must exercise nonzero current"
        np.testing.assert_allclose(
            j0,
            expected[direction],
            rtol=rtol(),
            atol=rtol() * scale,
            err_msg=f"J0{direction}: PEC current reflection",
        )
