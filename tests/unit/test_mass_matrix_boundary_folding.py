# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
# License: BSD-3-Clause-LBNL

"""Compare current folding with PC mass-matrix folding for cubic shapes.

Sum synthetic per-box coefficients, extend random E through its BCs, compute
J0 = sigma_pc E, and apply the current BCs. Fold the same matrix and compute
J1 using valid E columns only. These currents must agree on every valid row.
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

SHAPE_ORDER = 3


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


def _fill_cubic_pc(mf, direction, xp):
    """Randomize every coefficient supported by particles inside each box.

    A cubic shape covers four consecutive indices. For a particle inside
    a box's cells, the leftmost index ranges from cell_lo - 1 to cell_hi - 1
    for nodal data, or cell_lo - 2 to cell_hi - 1 for centered data. A row
    and its E column are supported when some such window contains both.
    This leaves unsupported ghost coefficients zero without depositing particles.
    """
    order = SHAPE_ORDER
    n_stencil = 2 * order + 1
    assert mf.n_comp == n_stencil**2
    for mfi in mf:
        valid = mfi.validbox()
        fab = mfi.fabbox()
        values = mf.array(mfi).to_xp()
        values[...] = 0.0
        rows = [
            np.arange(fab.small_end[axis], fab.big_end[axis] + 1) for axis in (0, 1)
        ]
        nodal = [int(mf.box_array().ix_type().node_centered(axis)) for axis in (0, 1)]
        left_lo = [valid.small_end[axis] - 2 + nodal[axis] for axis in (0, 1)]
        # valid_hi is cell_hi + 1 for nodal data, and cell_hi for centered data.
        left_hi = [valid.big_end[axis] - nodal[axis] - 1 for axis in (0, 1)]
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
    order = SHAPE_ORDER
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
        xhi = j.shape[0] if valid.big_end[0] == domain.big_end[0] else j.shape[0] - ng_x
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


def _valid_view(mf, mfi):
    """View a FAB's valid rows, retaining its component axis."""
    valid = mfi.validbox()
    rows = tuple(
        slice(
            mf.n_grow_vect[axis],
            mf.n_grow_vect[axis] + valid.big_end[axis] - valid.small_end[axis] + 1,
        )
        for axis in (0, 1)
    )
    return mf.array(mfi).to_xp()[rows[0], rows[1], 0, :]


def _apply_folded_pc(sigma, efield, current, n_cell, direction, xp):
    """Apply folded coefficients using global valid E, never physical E guards.

    Z columns wrap periodically, including the duplicate upper nodal plane.
    Coefficients pointing outside the physical X domain must be exactly zero.
    Check PEC row/column zeros independently of the matrix-vector product:
    E is already zero on constrained nodes and could otherwise hide an error.
    """
    order = SHAPE_ORDER
    n_stencil = 2 * order + 1
    assert sigma.n_comp == n_stencil**2
    e = xp.asarray(efield[...])
    current.set_val(0.0)
    for mfi in current:
        valid = mfi.validbox()
        rows = [
            np.arange(valid.small_end[axis], valid.big_end[axis] + 1) for axis in (0, 1)
        ]
        s = _valid_view(sigma, mfi)
        j = _valid_view(current, mfi)[:, :, 0]
        if direction != "x":
            constrained = xp.asarray((rows[0] == 0) | (rows[0] == n_cell[0]))
            assert not bool(xp.any(s[constrained, :, :] != 0.0)), (
                f"S{direction}{direction}: nonzero constrained PEC row"
            )
        for dz in range(-order, order + 1):
            column_z = xp.asarray((rows[1] + dz) % n_cell[1])
            for dx in range(-order, order + 1):
                column_x = rows[0] + dx
                inside = (column_x >= 0) & (column_x < e.shape[0])
                coefficient = dx + order + n_stencil * (dz + order)
                values = s[:, :, coefficient]
                assert not bool(xp.any(values[xp.asarray(~inside), :] != 0.0)), (
                    f"S{direction}{direction}: offset {(dx, dz)} retains physical E ghosts"
                )
                if direction != "x":
                    constrained = xp.asarray((column_x == 0) | (column_x == n_cell[0]))
                    assert not bool(xp.any(values[constrained, :] != 0.0)), (
                        f"S{direction}{direction}: offset {(dx, dz)} retains a PEC column"
                    )
                keep = xp.asarray(inside)
                ex = xp.asarray(column_x[inside])
                j[keep, :] += values[keep, :] * e[ex[:, None], column_z[None, :]]


def test_mass_matrix_boundary_folding_pec_x_periodic_z():
    """Folding the full cubic PC stencil must reproduce PEC folding of sigma E."""
    n_cell = [16, 16]
    sim = make_sim(
        n_cell=n_cell,
        max_grid_size=8,
        particle_shape="cubic",
        current_deposition_algo="direct",
    )
    grid = sim.solver.grid
    grid.lower_boundary_conditions = ["dirichlet", "periodic"]
    grid.upper_boundary_conditions = ["dirichlet", "periodic"]
    grid.lower_boundary_conditions_particles = ["absorbing", "periodic"]
    grid.upper_boundary_conditions_particles = ["absorbing", "periodic"]
    sim.evolve_scheme = picmi.ThetaImplicitEMEvolveScheme(
        nonlinear_solver=picmi.NewtonNonlinearSolver(
            linear_solver=picmi.GMRESLinearSolver(),
            pc_type=picmi.JacobiPreconditioner(),
            use_mass_matrices_jacobian=True,
            use_mass_matrices_pc=True,
            mass_matrices_pc_width=SHAPE_ORDER,
        ),
        theta=0.5,
    )
    # An empty species enables particle gather/deposition guard-cell allocation.
    sim.add_species(
        picmi.Species(particle_type="electron", name="electrons"), layout=None
    )
    sim.initialize_inputs()
    sim.initialize_warpx()

    warpx = sim.extension.warpx
    fields = sim.fields
    xp = _array_module()
    for idir, direction in enumerate(("x", "y", "z")):
        sigma = fields.get("MassMatrices_PC", direction, 0)
        _fill_cubic_pc(sigma, idir, xp)
        template = fields.get("current_fp", direction, 0)
        for name in ("J0", "J1"):
            fields.alloc_init(
                name,
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

    # Fold the same already-summed coefficients in place. Cartesian geometry
    # needs no radial volume scaling. Do not sum or apply current BCs to J1.
    warpx.apply_mass_matrices_pc_boundary()
    for direction in ("x", "y", "z"):
        j0 = fields.get("J0", direction, 0)
        j1 = fields.get("J1", direction, 0)
        _apply_folded_pc(
            fields.get("MassMatrices_PC", direction, 0),
            fields.get("Efield_fp", direction, 0),
            j1,
            n_cell,
            direction,
            xp,
        )
        scale = np.max(np.abs(j0[...]))
        # Check every FAB's valid rows, including all copies of shared nodes.
        for mfi in j1:
            reference = _valid_view(j0, mfi)
            actual = _valid_view(j1, mfi)
            assert bool(xp.all(xp.isfinite(actual))), (
                f"J1{direction}: non-finite current"
            )
            error = xp.abs(actual - reference)
            tolerance = rtol() * (scale + xp.abs(reference))
            mismatches = xp.argwhere(~(error <= tolerance))
            if len(mismatches):
                valid = mfi.validbox()
                lo = tuple(valid.small_end[axis] for axis in (0, 1))
                hi = tuple(valid.big_end[axis] for axis in (0, 1))
                lines = [
                    f"J1{direction} != J0{direction}: valid box {lo} to {hi}",
                    f"mismatches={len(mismatches)}/{actual.size}, "
                    f"max_abs_error={float(xp.max(error)):.16e}",
                    "Global grid indices; showing up to 20 mismatches:",
                    "ix iz comp J0 J1 abs_error tolerance",
                ]
                for index in mismatches[:20]:
                    ix, iz, comp = (int(value) for value in index)
                    key = (ix, iz, comp)
                    lines.append(
                        f"{lo[0] + ix} {lo[1] + iz} {comp} "
                        f"{float(reference[key]):.16e} {float(actual[key]):.16e} "
                        f"{float(error[key]):.16e} {float(tolerance[key]):.16e}"
                    )
                pytest.fail("\n".join(lines), pytrace=False)
