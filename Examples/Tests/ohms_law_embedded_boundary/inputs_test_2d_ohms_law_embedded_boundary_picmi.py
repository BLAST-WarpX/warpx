#!/usr/bin/env python3
#
# --- 2D (XZ) unit tests of the hybrid solver's embedded-boundary value
# --- application (the 2D counterpart of
# --- inputs_test_3d_ohms_law_embedded_boundary_picmi.py): a small (x, z)
# --- grid with a deterministic conducting wall is initialized, synthetic
# --- fields with known closed-form behavior are loaded through the field
# --- wrappers, the boundary operators are applied directly through their
# --- Python bindings, and the values are asserted point by point.
# ---
# --- Planar wall (conductor x > X_WALL, fluid normal -x_hat): Ey (the
# --- out-of-plane component, purely tangential in 2D) and Ez are TANGENTIAL
# --- (odd mirrors), Ex is NORMAL (even); for B the parity is swapped (Bx
# --- odd, By/Bz even). All normals are axis aligned, so the closed forms
# --- hold to round-off.
# ---
# --- Staggered runs use the STAIRCASE update masks
# --- (MarkUpdateCellsStairCase): a staggered point adjacent to ANY partially
# --- cut cell is unowned (eb_update = 0) and becomes a level-set mirror-fill
# --- target. The wall at node 24.3 cuts cell 24, so exactly as in 3D:
# ---   * nodal-x rows (Ey/Ez/Bx) with i <= 23 (s >= +1.3h) stay owned; the
# ---     first FLUID row at s = +0.3h is a direct fill target (offset = h,
# ---     image exactly at the first owned row, ratio s/(s+h)) and the
# ---     covered row at s = -0.7h is ill-posed (0.6 of its image weight sits
# ---     on the reclassified s = +0.3h row < W_MIN = 0.5) and is filled by
# ---     the cascade -- both land on the same odd line c*s/(s_first+h),
# ---   * the cut cell-centered-x rows (s = -0.2h) are mirror-fill targets
# ---     for Ex (even normal) and By/Bz (even tangential) alike,
# ---   * the mask-based J folds only write OWNED rows: the s = +1.3h row
# ---     receives -0.4c and the unowned s = +0.3h row is skipped; the
# ---     level-set-based rho fold still writes both fluid rows
# ---     (-0.6c / -0.4c, conserving the folded amount).

import argparse
import sys

import numpy as np

from pywarpx import fields, picmi

constants = picmi.constants

N_X = 32
N_Z = 32
LO = -1.0
HI = 1.0
H = (HI - LO) / N_X  # square cells: dx = dz = H

# planar wall: conductor x > X_WALL, fluid-pointing normal -x_hat; the 0.3 h
# offset from a node puts a cut x-edge with a covered center next to the wall
X_WALL_OFFSET = 0.3
X_WALL_NODE = 24
X_WALL = LO + (X_WALL_NODE + X_WALL_OFFSET) * H

# mirror geometry of the fill operators (EBJBoundary.cpp)
D_IMG_MIN = 0.5 * H


def vector_ratio(s):
    """Odd-parity scaling s/d_im of the staggered vector fill."""
    offset = max(max(abs(s), D_IMG_MIN) - s, H)
    return s / (s + offset)


def scalar_ratio(s):
    """Odd-parity scaling s/d_im of the nodal scalar fill (exact mirror)."""
    return s / max(abs(s), D_IMG_MIN)


def setup_simulation(collocated=False):
    grid = picmi.Cartesian2DGrid(
        number_of_cells=[N_X, N_Z],
        lower_bound=[LO, LO],
        upper_bound=[HI, HI],
        lower_boundary_conditions=["dirichlet", "periodic"],
        upper_boundary_conditions=["dirichlet", "periodic"],
        lower_boundary_conditions_particles=["absorbing", "periodic"],
        upper_boundary_conditions_particles=["absorbing", "periodic"],
        warpx_max_grid_size=2048,
        # split along x (the wall-normal direction) for a meaningful 2-rank
        # run; the box seam (i=16) is 8 cells from the wall (i=24.3), clear
        # of every fill/fold band
        warpx_max_grid_size_x=2048 if collocated else N_X // 2,
        warpx_blocking_factor=8,
    )

    sim = picmi.Simulation(
        time_step_size=1.0e-9,
        max_steps=1,
        particle_shape=1,
        verbose=0,
    )
    sim.current_deposition_algo = "direct"
    sim.grid_type = "collocated" if collocated else "staggered"

    # The wall cuts cell 24 (node 24.3): the staircase masks make the
    # planar-wall closed forms identical to the 3D battery
    sim.solver = picmi.HybridPICSolver(
        grid=grid,
        gamma=5.0 / 3.0,
        Te=0.0,
        n0=1.0e18,
        n_floor=1.0e16,
        plasma_resistivity=1.0e-6,
        substeps=4,
        # The staggered battery deliberately exercises the always-on staircase
        # EB fills (its gates assert that path), so the conformal wall
        # treatment is only enabled on the collocated battery here.
        use_conformal_eb=True if collocated else None,
    )

    sim.embedded_boundary = picmi.EmbeddedBoundary(
        implicit_function="(x-xw)", xw=X_WALL
    )

    sim.initialize_inputs()
    sim.initialize_warpx()
    return sim


class CheckSet:
    """Collect named assertions and report them pytest-style."""

    def __init__(self):
        self.failures = []

    def expect(self, name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
        if not condition:
            self.failures.append(f"{name}: {detail}")

    def close(self, name, a, b, tol, label=""):
        err = np.max(np.abs(np.asarray(a) - np.asarray(b)))
        self.expect(name, err <= tol, f"max|err|={err:.3e} tol={tol:.1e} {label}")

    def finish(self):
        n = len(self.failures)
        print(f"\n{'all checks passed' if n == 0 else f'{n} CHECKS FAILED'}")
        assert n == 0, "\n".join(self.failures)


def arr2d(field):
    """Valid-domain (x, z) view of a field wrapper or raw global array,
    dropping a trailing single-component axis if one is exposed."""
    a = np.asarray(field[...])
    return a[:, :, 0] if a.ndim == 3 else a


def set_xprofile(wrapper, prof_x):
    """Load a pure-x profile, uniform in z (and in the component axis)."""
    shape = np.asarray(wrapper[...]).shape
    prof = prof_x[:, None] if len(shape) == 2 else prof_x[:, None, None]
    wrapper[...] = np.broadcast_to(prof, shape).copy()


def set_xrows(wrapper, rows, value):
    """Overwrite the given global x rows of a wrapper with a value."""
    a = np.array(np.asarray(wrapper[...]))
    a[rows, ...] = value
    wrapper[...] = a


# ----------------------------------------------------------------------------
# Planar-wall battery (round-off accurate)
# ----------------------------------------------------------------------------
def run_plane_battery(sim):
    wx = sim.extension.warpx
    ck = CheckSet()
    c = 2.0  # constant field amplitude
    b = 3.0  # linear field slope (per unit s)

    # staggered x coordinates and signed distances s = X_WALL - x
    x_node = LO + np.arange(N_X + 1) * H
    x_cent = LO + (np.arange(N_X) + 0.5) * H
    s_node = X_WALL - x_node
    s_cent = X_WALL - x_cent

    def node_rows(lo, hi):
        return [i for i in range(N_X + 1) if lo < s_node[i] / H <= hi]

    def cent_rows(lo, hi):
        return [i for i in range(N_X) if lo < s_cent[i] / H <= hi]

    Ex = fields.ExFPWrapper()  # (cc x, nodal z): normal to the x wall
    Ey = fields.EyFPWrapper()  # (nodal x, nodal z): out-of-plane, tangential
    Ez = fields.EzFPWrapper()  # (nodal x, cc z): tangential to the x wall
    Bx = fields.BxFPWrapper()  # (nodal x, cc z): normal -> odd fill
    By = fields.ByFPWrapper()  # (cc x, cc z): in-plane flux face (ECT-owned)
    Bz = fields.BzFPWrapper()  # (cc x, nodal z): tangential, on x-edges
    rho = fields.RhoFPWrapper()  # nodal (x, z)
    pe = fields.ElectronPressureFPWrapper()  # nodal (x, z)

    # --- 0) 2D wrapper shapes encode the Yee staggering; everything
    # --- downstream indexes (x, z) with x = axis 0
    expected = {
        "Ex (cc x, nodal z)": (Ex, (N_X, N_Z + 1)),
        "Ey (nodal x, nodal z)": (Ey, (N_X + 1, N_Z + 1)),
        "Ez (nodal x, cc z)": (Ez, (N_X + 1, N_Z)),
        "Bx (nodal x, cc z)": (Bx, (N_X + 1, N_Z)),
        "By (cc x, cc z)": (By, (N_X, N_Z)),
        "Bz (cc x, nodal z)": (Bz, (N_X, N_Z + 1)),
        "rho (nodal x, nodal z)": (rho, (N_X + 1, N_Z + 1)),
        "Pe (nodal x, nodal z)": (pe, (N_X + 1, N_Z + 1)),
    }
    for name, (w, shape) in expected.items():
        got = arr2d(w).shape
        full = np.asarray(w[...]).shape
        ck.expect(f"staggering: {name} -> {shape}", got == shape, f"shape={full}")

    # staircase-mask row layout on the nodal-x staggerings (Ey, Ez, Bx):
    # i <= 23 (s >= +1.3h) solver-owned; i = 24 (s = +0.3h) unowned FLUID ->
    # direct fill target (offset = h, image exactly at node 23, ratio
    # s/(s+h)); i = 25 (s = -0.7h) covered and ill-posed -> cascade fill from
    # the locked node-23/24 values; i >= 26 deep (zeroed)
    i_owned = node_rows(1.0, 100.0)  # solver-owned node rows (s >= +1.3h)
    i_first = node_rows(0.05, 0.5)  # first unowned fluid row, s = +0.3h
    i_band = node_rows(-1.0, -0.05)  # covered cascade row, s = -0.7h
    i_deep = node_rows(-100.0, -1.0)
    i_cut = cent_rows(-0.5, 0.0)  # cut x-edge with covered center, s = -0.2h

    # Both unowned nodal-x rows land on the same odd line for a constant
    # tangential (or normal-B) field c: the direct fill of the first row has
    # d_im = s_first + h and the cascade fill of the covered row linearly
    # interpolates the locked node-23/24 values, i.e. the same line c*s/d_im
    # pinned to 0 at the wall.
    s_first = s_node[X_WALL_NODE]

    def odd_line(s):
        return s / (s_first + H)

    # --- 1) tangential E (Ey out-of-plane, Ez in-plane): odd mirror --------
    Ex[...] = 0.0
    Ey[...] = c
    Ez[...] = c
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    ey = arr2d(Ey)
    ez = arr2d(Ez)

    ck.close("E tangential: Ey owned rows untouched", ey[i_owned, :], c, 0.0)
    ck.close("E tangential: Ez owned rows untouched", ez[i_owned, :], c, 0.0)
    for i in i_first:  # first unowned fluid row: direct mirror fill
        ck.close(
            f"E tangential: Ey unowned fluid row mirror-filled at s={s_node[i] / H:+.2f}h",
            ey[i, :],
            vector_ratio(s_node[i]) * c,
            1e-12,
        )
        ck.close(
            f"E tangential: Ez unowned fluid row mirror-filled at s={s_node[i] / H:+.2f}h",
            ez[i, :],
            vector_ratio(s_node[i]) * c,
            1e-12,
        )
    for i in i_band:  # covered row: cascade fill from the locked rows
        ck.close(
            f"E tangential: Ey cascade odd mirror at s={s_node[i] / H:+.2f}h",
            ey[i, :],
            odd_line(s_node[i]) * c,
            1e-12,
        )
        ck.close(
            f"E tangential: Ez cascade odd mirror at s={s_node[i] / H:+.2f}h",
            ez[i, :],
            odd_line(s_node[i]) * c,
            1e-12,
        )
    ck.close("E tangential: Ey deep interior zero", ey[i_deep, :], 0.0, 0.0)
    ck.close("E tangential: Ez deep interior zero", ez[i_deep, :], 0.0, 0.0)

    # --- 2) tangential E: linear continuation through the surface ----------
    # Ey = Ez = b*s in the fluid must continue as b*s through both fill rows
    # (the bilinear gather is exact on linear fields, and the cascade
    # interpolates exact linear values): Dirichlet 0 at the wall
    set_xprofile(Ey, b * s_node)
    set_xprofile(Ez, b * s_node)
    Ex[...] = 0.0
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    ey = arr2d(Ey)
    ez = arr2d(Ez)
    for i in i_first + i_band:
        ck.close(
            f"E tangential: Ey linear field continues b*s at s={s_node[i] / H:+.2f}h",
            ey[i, :],
            b * s_node[i],
            1e-12,
        )
        ck.close(
            f"E tangential: Ez linear field continues b*s at s={s_node[i] / H:+.2f}h",
            ez[i, :],
            b * s_node[i],
            1e-12,
        )

    # --- 3) normal E (Ex): even mirror + covered-center cut edge -----------
    Ex[...] = c
    Ey[...] = 0.0
    Ez[...] = 0.0
    # poke junk into the cut x-edge row whose center is covered
    # (edge [x_24, x_25] contains the wall; center s = (0.3-0.5)h = -0.2h)
    set_xrows(Ex, i_cut, 1.0e6)
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    ex = arr2d(Ex)
    ck.close("E normal: fluid rows untouched", ex[cent_rows(0.05, 100.0), :], c, 0.0)
    for i in i_cut:
        ck.close(
            f"E normal: covered-center cut edge filled (junk replaced) at s={s_cent[i] / H:+.2f}h",
            ex[i, :],
            c,
            1e-12,
        )
    ck.close("E normal: deep interior zero", ex[cent_rows(-100.0, -1.0), :], 0.0, 0.0)

    # --- 4) B: swapped parity (normal Bx odd, tangential By/Bz even) -------
    # Bx lives on the nodal-x rows, so the odd closed forms match Ey/Ez; the
    # cut By face and the cut Bz x-edge (centers at s = -0.2h) are unowned
    # under the staircase marking and are mirror-filled with the even
    # tangential parity: poke junk in to prove they are actively refilled
    Bx[...] = c
    By[...] = c
    Bz[...] = c
    set_xrows(By, i_cut, 1.0e6)
    set_xrows(Bz, i_cut, 1.0e6)
    wx.hybrid_apply_eb_boundary_to_face_field("Bfield_fp", 0)
    bx = arr2d(Bx)
    by = arr2d(By)
    bz = arr2d(Bz)
    ck.close("B normal: owned rows untouched", bx[i_owned, :], c, 0.0)
    for i in i_first:
        ck.close(
            f"B normal: unowned fluid row mirror-filled at s={s_node[i] / H:+.2f}h",
            bx[i, :],
            vector_ratio(s_node[i]) * c,
            1e-12,
        )
    for i in i_band:
        ck.close(
            f"B normal: cascade odd mirror at s={s_node[i] / H:+.2f}h",
            bx[i, :],
            odd_line(s_node[i]) * c,
            1e-12,
        )
    ck.close("B normal: deep zero", bx[i_deep, :], 0.0, 0.0)
    ck.close(
        "B in-plane flux: By fluid rows untouched",
        by[cent_rows(0.05, 100.0), :],
        c,
        0.0,
    )
    ck.close(
        "B tangential: Bz fluid rows untouched", bz[cent_rows(0.05, 100.0), :], c, 0.0
    )
    for i in i_cut:
        ck.close(
            f"B in-plane flux: cut By face even mirror-filled (junk replaced) at s={s_cent[i] / H:+.2f}h",
            by[i, :],
            c,
            1e-12,
        )
        ck.close(
            f"B tangential: cut Bz x-edge even mirror-filled (junk replaced) at s={s_cent[i] / H:+.2f}h",
            bz[i, :],
            c,
            1e-12,
        )
    ck.close("B in-plane flux: By deep zero", by[cent_rows(-100.0, -1.0), :], 0.0, 0.0)
    ck.close("B tangential: Bz deep zero", bz[cent_rows(-100.0, -1.0), :], 0.0, 0.0)

    # --- 5) nodal scalars: rho odd (Dirichlet 0), Pe even (Neumann) --------
    # the scalar fill is keyed off the level set alone (not the staircase
    # masks), so every fluid row -- including the unowned s = +0.3h row -- is
    # untouched
    i_fluid_nodes = node_rows(0.05, 100.0)
    rho[...] = c
    wx.hybrid_apply_eb_boundary_to_nodal_scalar("rho_fp", 0, True)
    r = arr2d(rho)
    ck.close("rho odd: fluid untouched", r[i_fluid_nodes, :], c, 0.0)
    for i in i_band:
        ck.close(
            f"rho odd: mirror at s={s_node[i] / H:+.2f}h",
            r[i, :],
            scalar_ratio(s_node[i]) * c,
            1e-12,
        )
    ck.close("rho odd: deep zero", r[i_deep, :], 0.0, 0.0)

    # linear rho = b*s: ghosts continue b*s exactly (zero at the surface)
    set_xprofile(rho, b * s_node)
    wx.hybrid_apply_eb_boundary_to_nodal_scalar("rho_fp", 0, True)
    r = arr2d(rho)
    for i in i_band:
        ck.close(
            f"rho odd: linear field continues b*s at s={s_node[i] / H:+.2f}h",
            r[i, :],
            b * s_node[i],
            1e-12,
        )

    pe[...] = c
    wx.hybrid_apply_eb_boundary_to_nodal_scalar("hybrid_electron_pressure_fp", 0, False)
    p = arr2d(pe)
    ck.close("Pe even: fluid untouched", p[i_fluid_nodes, :], c, 0.0)
    for i in i_band:
        ck.close(
            f"Pe even: Neumann mirror at s={s_node[i] / H:+.2f}h", p[i, :], c, 1e-12
        )
    ck.close("Pe even: deep zero", p[i_deep, :], 0.0, 0.0)

    # --- 6) deposit fold: PEC image parities, planar closed forms ----------
    # The covered node row at s=-0.7h holds a deposit c (the shape-function
    # spill of wall-adjacent particles). Its mirror lands 0.6 of the way
    # between the first two fluid rows. The level-set-based rho fold writes
    # both fluid rows (-0.6c and -0.4c, PEC image charge subtracted,
    # conserving the folded amount); the mask-based J fold only writes
    # S_SOLUTION rows, so the unowned s=+0.3h row is SKIPPED (it is a fill
    # target, overwritten by the fill anyway) and only the first owned row
    # receives its -0.4c share.
    i_dep = i_band  # s = -0.7h (single covered row for this wall)
    i_second = node_rows(1.0, 1.5)  # s = +1.3h (first owned row)
    rho[...] = 0.0
    set_xrows(rho, i_dep, c)
    wx.hybrid_fold_eb_deposit_to_nodal_scalar("rho_fp", 0)
    r = arr2d(rho)
    ck.close("fold rho: first fluid row receives -0.6c", r[i_first, :], -0.6 * c, 1e-12)
    ck.close(
        "fold rho: second fluid row receives -0.4c", r[i_second, :], -0.4 * c, 1e-12
    )
    ck.close(
        "fold rho: fluid beyond the fold reach untouched",
        r[node_rows(1.6, 100.0), :],
        0.0,
        0.0,
    )
    ck.close(
        "fold rho: covered deposit left in place for the fill", r[i_dep, :], c, 0.0
    )
    ck.close(
        "fold rho: folded amount conserved (sum = -c)",
        r[i_first[0], :] + r[i_second[0], :],
        -c,
        1e-12,
    )

    # tangential J (Jy out-of-plane and Jz in-plane, no cross-talk for an
    # x-normal wall): image current antiparallel (subtracted), but only the
    # owned row is a fold target under the staircase masks
    Jx = fields.JxFPWrapper()
    Jy = fields.JyFPWrapper()
    Jz = fields.JzFPWrapper()
    Jx[...] = 0.0
    Jy[...] = 0.0
    Jz[...] = 0.0
    set_xrows(Jy, i_dep, c)
    set_xrows(Jz, i_dep, c)
    wx.hybrid_fold_eb_deposit_to_edge_field("current_fp", 0)
    jy = arr2d(Jy)
    jz = arr2d(Jz)
    ck.close(
        "fold Jy tangential: unowned fluid row skipped (fill target)",
        jy[i_first, :],
        0.0,
        0.0,
    )
    ck.close(
        "fold Jy tangential: first owned row subtracted -0.4c",
        jy[i_second, :],
        -0.4 * c,
        1e-12,
    )
    ck.close(
        "fold Jz tangential: unowned fluid row skipped (fill target)",
        jz[i_first, :],
        0.0,
        0.0,
    )
    ck.close(
        "fold Jz tangential: first owned row subtracted -0.4c",
        jz[i_second, :],
        -0.4 * c,
        1e-12,
    )
    ck.close(
        "fold J tangential: covered deposit left in place for the fill",
        jy[i_dep, :],
        c,
        0.0,
    )

    # normal J (Jx): image current parallel (added); the deposit sits on the
    # cut x-edge with covered center (s=-0.2h), its mirror lands 0.6 of the
    # way between the first two fluid x-edge rows
    Jx[...] = 0.0
    Jy[...] = 0.0
    Jz[...] = 0.0
    set_xrows(Jx, i_cut, c)
    wx.hybrid_fold_eb_deposit_to_edge_field("current_fp", 0)
    jx = arr2d(Jx)
    i_cn = cent_rows(0.5, 1.0)  # s = +0.8h
    ck.close("fold J normal: added +0.4c", jx[i_cn, :], 0.4 * c, 1e-12)
    ck.close(
        "fold J normal: fluid beyond the fold reach untouched",
        jx[cent_rows(1.6, 100.0), :],
        0.0,
        0.0,
    )

    # reflecting-wall parity: the exact opposite signs (deposit added back,
    # mass conserving; normal J subtracted)
    rho[...] = 0.0
    set_xrows(rho, i_dep, c)
    wx.hybrid_fold_eb_deposit_to_nodal_scalar("rho_fp", 0, pec=False)
    r = arr2d(rho)
    ck.close(
        "fold rho (reflect): first fluid row receives +0.6c",
        r[i_first, :],
        0.6 * c,
        1e-12,
    )
    ck.close(
        "fold rho (reflect): second fluid row receives +0.4c",
        r[i_second, :],
        0.4 * c,
        1e-12,
    )
    ck.close(
        "fold rho (reflect): folded amount conserved (sum = +c)",
        r[i_first[0], :] + r[i_second[0], :],
        c,
        1e-12,
    )

    Jx[...] = 0.0
    Jy[...] = 0.0
    Jz[...] = 0.0
    set_xrows(Jy, i_dep, c)
    wx.hybrid_fold_eb_deposit_to_edge_field("current_fp", 0, pec=False)
    jy = arr2d(Jy)
    ck.close(
        "fold Jy tangential (reflect): unowned fluid row skipped",
        jy[i_first, :],
        0.0,
        0.0,
    )
    ck.close(
        "fold Jy tangential (reflect): first owned row added +0.4c",
        jy[i_second, :],
        0.4 * c,
        1e-12,
    )
    Jx[...] = 0.0
    Jy[...] = 0.0
    Jz[...] = 0.0
    set_xrows(Jx, i_cut, c)
    wx.hybrid_fold_eb_deposit_to_edge_field("current_fp", 0, pec=False)
    jx = arr2d(Jx)
    ck.close("fold J normal (reflect): subtracted -0.4c", jx[i_cn, :], -0.4 * c, 1e-12)

    # --- 7) selectivity against a spatially varying field ------------------
    shape = np.asarray(Ey[...]).shape
    idx = np.indices(shape)
    varying = 1.0 + 0.17 * idx[0] + 0.013 * idx[1]
    Ey[...] = varying
    Ex[...] = 0.0
    Ez[...] = 0.0
    before = np.array(np.asarray(Ey[...]))
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    after = np.asarray(Ey[...])
    ck.close(
        "selectivity: varying field bit-identical in the owned fluid",
        after[i_owned, :],
        before[i_owned, :],
        0.0,
    )
    # the unowned fluid row is rewritten with the mirror ratio of the value
    # at its image point (exactly the first owned row for this wall)
    for i in i_first:
        ck.close(
            f"selectivity: unowned fluid row rewritten with the mirror ratio at s={s_node[i] / H:+.2f}h",
            after[i, :],
            vector_ratio(s_node[i]) * before[i - 1, :],
            1e-12,
        )

    ck.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--battery",
        choices=["eb"],
        default="eb",
        help="which unit battery to run (the isotropized-stencil batteries "
        "moved to Examples/Tests/ohm_solver_isotropic_operators with the "
        "standalone isotropic-operators PR)",
    )
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    sim = setup_simulation()
    run_plane_battery(sim)


if __name__ == "__main__":
    main()
