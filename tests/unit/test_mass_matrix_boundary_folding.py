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


def _apply_unfolded_pc(sigma, efield, current, boundary_axes, xp):
    """Direct diagonal-block stencil sum on valid and guard current rows.

    Array4 views retain the unused third spatial axis, so the XZ plane is
    [:, :, 0, component]. Component offsets flatten with X varying fastest.
    Reads are restricted to the E FAB, as in ApplyMassMatrices. Outer internal
    J guards need not have a complete stencil. Check that no nonzero term is
    clipped on valid rows or physical ghost rows needed for the reference.
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
        ng = current.n_grow_vect
        required_rows = [slice(ng[a], j.shape[a] - ng[a]) for a in (0, 1)]
        for a in boundary_axes:
            lo = 0 if valid.small_end[a] == domain.small_end[a] else ng[a]
            hi = (
                j.shape[a] if valid.big_end[a] == domain.big_end[a] else j.shape[a] - ng[a]
            )
            required_rows[a] = slice(lo, hi)
        required = np.zeros(j.shape, dtype=bool)
        required[tuple(required_rows)] = True
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


def _boundary_reference(mf, raw_current, n_cell, direction, pec_nodes):
    """Expected valid current after adding signed physical guard contributions."""
    expected = raw_current.copy()
    ng = mf.n_grow_vect
    size = [len(mf.imesh(axis)) for axis in (0, 1)]
    for axis, nodes in pec_nodes.items():
        # Retain transverse guards so the next direction includes corner terms.
        view = np.swapaxes(expected, 0, axis)
        nodal = int(mf.box_array().ix_type().node_centered(axis))
        for boundary in (0, n_cell[axis]):
            sign = 1.0 if direction == "xz"[axis] else -1.0
            if boundary not in nodes:  # PMC reverses PEC current parity.
                sign = -sign
            for i in range(size[axis]):
                mirror = 2 * boundary - (1 - nodal) - i
                if 0 <= mirror < size[axis] and mirror != i:
                    continue
                if -ng[axis] <= mirror < size[axis] + ng[axis]:
                    # At a PMC tangential boundary node, mirror == i doubles J.
                    view[i + ng[axis], :] += sign * view[mirror + ng[axis], :]
    return expected[tuple(slice(ng[a], ng[a] + size[a]) for a in (0, 1))]


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


def _apply_folded_pc(sigma, efield, current, n_cell, direction, pec_nodes, xp):
    """Apply folded coefficients using global valid E, never physical E guards.

    Columns wrap only in periodic directions, including the upper nodal plane.
    Coefficients pointing outside the physical domain must be exactly zero.
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
        for axis, nodes in pec_nodes.items():
            if direction != "xz"[axis]:
                constrained = xp.asarray(np.isin(rows[axis], nodes))
                assert not bool(xp.any(xp.swapaxes(s, 0, axis)[constrained] != 0.0)), (
                    f"S{direction}{direction}: nonzero constrained PEC row"
                )
        for dz in range(-order, order + 1):
            for dx in range(-order, order + 1):
                columns = [rows[0] + dx, rows[1] + dz]
                coefficient = dx + order + n_stencil * (dz + order)
                values = s[:, :, coefficient]
                keep = []
                for axis, column in enumerate(columns):
                    if axis not in pec_nodes:
                        column %= n_cell[axis]
                    inside = (column >= 0) & (column < e.shape[axis])
                    v = xp.swapaxes(values, 0, axis)
                    assert not bool(xp.any(v[xp.asarray(~inside)] != 0.0)), (
                        f"S{direction}{direction}: offset {(dx, dz)} retains physical E ghosts"
                    )
                    if direction != "xz"[axis]:
                        constrained = xp.asarray(np.isin(column, pec_nodes.get(axis, [])))
                        assert not bool(xp.any(v[constrained] != 0.0)), (
                            f"S{direction}{direction}: offset {(dx, dz)} retains a PEC column"
                        )
                    keep.append(xp.asarray(inside))
                row_index = xp.ix_(*(xp.nonzero(k)[0] for k in keep))
                column_index = xp.ix_(
                    *(xp.asarray(columns[a])[keep[a]] for a in (0, 1))
                )
                j[row_index] += values[row_index] * e[column_index]


@pytest.mark.parametrize(
    "boundary_axes, pmc_lo",
    [
        pytest.param((0,), False, id="pec_x_periodic_z"),
        pytest.param((1,), False, id="periodic_x_pec_z"),
        pytest.param((0,), True, id="pmc_lo_x_pec_hi_x_periodic_z"),
        pytest.param((0, 1), True, id="pmc_lo_x_pec_hi_x_pec_z"),
    ],
)
def test_mass_matrix_boundary_folding(boundary_axes, pmc_lo):
    """Folding the full cubic PC stencil must reproduce current BCs on sigma E."""
    n_cell = [16, 16]
    # Each key is a physical boundary axis; values list its constrained PEC faces.
    pec_nodes = {axis: [0, n_cell[axis]] for axis in boundary_axes}
    if pmc_lo:
        pec_nodes[boundary_axes[0]].remove(0)
    sim = make_sim(
        n_cell=n_cell,
        max_grid_size=8,
        particle_shape="cubic",
        current_deposition_algo="direct",
    )
    grid = sim.solver.grid
    for bc in (grid.lower_boundary_conditions, grid.upper_boundary_conditions):
        for axis in boundary_axes:
            bc[axis] = "dirichlet"
    for bc in (
        grid.lower_boundary_conditions_particles,
        grid.upper_boundary_conditions_particles,
    ):
        for axis in boundary_axes:
            bc[axis] = "absorbing"
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
    if pmc_lo:
        # PICMI has no PMC mapping; override the translated WarpX input.
        pywarpx.boundary.field_lo[boundary_axes[0]] = "pmc"
    # This test sums the PC matrices before applying them to E. Contributions
    # from neighboring boxes extend the cubic stencil to three cells beyond a
    # box edge. Applying box-local matrices first and then summing J only needs
    # the local particle-gather support (see the discussion in PR #7254):
    # https://github.com/BLAST-WarpX/warpx/pull/7254#discussion_r4030996489
    # E already has enough allocated guards, but the normal boundary fill covers
    # only two cells. Fill all allocated guards so J0 also reads valid E values
    # where physical guards intersect transverse box/periodic boundaries.
    pywarpx.warpx.safe_guard_cells = 1
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
    for axis, nodes in pec_nodes.items():
        for direction in (d for d in ("x", "y", "z") if d != "xz"[axis]):
            e = fields.get("Efield_fp", direction, 0)[...]
            assert np.all(np.take(e, nodes, axis=axis) == 0.0), (
                f"E{direction}: PEC Dirichlet nodes"
            )

    expected = {}
    for direction in ("x", "y", "z"):
        j0 = fields.get("J0", direction, 0)
        _apply_unfolded_pc(
            fields.get("MassMatrices_PC", direction, 0),
            fields.get("Efield_fp", direction, 0),
            j0,
            boundary_axes,
            xp,
        )
        expected[direction] = _boundary_reference(
            j0, j0[()], n_cell, direction, pec_nodes
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
            err_msg=f"J0{direction}: current boundary reflection",
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
            pec_nodes,
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
