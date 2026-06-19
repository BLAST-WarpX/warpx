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
# --- odd; the cut By/Bz points are solver-owned). All normals are axis
# --- aligned, so the closed forms hold to round-off.
# ---
# --- Unlike RZ (staircase masks), this deck runs with
# --- hybrid_pic_model.use_conformal_eb = 1, which was just unguarded for
# --- 2D XZ. The conformal (ECT) masks keep every staggered point whose
# --- edge/face is only partially cut solver-owned (eb_update = 1), exactly
# --- as in 3D, so the 3D closed forms carry over unchanged:
# ---   * the first fluid rows (s = +0.3h) stay untouched by the fills and
# ---     RECEIVE the deposit folds,
# ---   * the only vector fill band row is the covered row at s = -0.7h
# ---     (direct mirror, ratio s/d_im = -1 against constants),
# ---   * the cut x-edge with covered center (s = -0.2h) is an Ex fill
# ---     target (fill_covered_centers) but the cut By face / Bz x-edge is
# ---     left to the conformal update.
# ---
# --- The Marder battery ports the 3D transitional-Marder unit tests with
# --- the 2D nodal divergence d(Ex)/dx + d(Ez)/dz (no y term). In 2D the
# --- correction updates Ex and Ez ONLY; the out-of-plane Ey has no
# --- divergence contribution and must come through a Marder call
# --- bit-identical (asserted).

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


def setup_simulation(hyper=False, resistive=False, collocated=False):
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
        # single box for the hyper battery: the species-free deck allocates
        # the plasma current with zero ghost cells, so the hyper-resistivity
        # stencil must not straddle a box seam
        warpx_max_grid_size_x=2048 if (hyper or resistive or collocated) else N_X // 2,
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

    # use_conformal_eb was just unguarded for 2D XZ: the ECT masks make the
    # planar-wall closed forms identical to the 3D battery
    sim.solver = picmi.HybridPICSolver(
        grid=grid,
        gamma=5.0 / 3.0,
        Te=0.0,
        n0=1.0e18,
        n_floor=1.0e16,
        # the hyper battery isolates the hyper-resistive term
        # (E = -eta_h*nabla^2 J); the resistive battery isolates the resistive
        # term (E = eta*J) with the corner-curl isotropization on.
        plasma_resistivity=(1.0 if resistive else (0.0 if hyper else 1.0e-6)),
        plasma_hyper_resistivity=1.0 if hyper else None,
        isotropic_resistivity=resistive,
        substeps=4,
        use_conformal_eb=True,
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

    # conformal-mask row layout on the nodal-x staggerings (Ey, Ez, Bx):
    # i <= 24 (s >= +0.3h) solver-owned; i = 25 (s = -0.7h) fill band;
    # i >= 26 deep. The s = +0.3h row is NOT a fill target (3D-like; the
    # staircase masks of RZ would make it one).
    i_fluid = node_rows(0.05, 100.0)  # all fluid node rows, incl. s = +0.3h
    i_band = node_rows(-1.0, -0.05)  # covered fill row, s = -0.7h
    i_deep = node_rows(-100.0, -1.0)
    i_cut = cent_rows(-0.5, 0.0)  # cut x-edge with covered center, s = -0.2h

    # --- 1) tangential E (Ey out-of-plane, Ez in-plane): odd mirror --------
    Ex[...] = 0.0
    Ey[...] = c
    Ez[...] = c
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    ey = arr2d(Ey)
    ez = arr2d(Ez)

    ck.close("E tangential: Ey fluid rows untouched", ey[i_fluid, :], c, 0.0)
    ck.close("E tangential: Ez fluid rows untouched", ez[i_fluid, :], c, 0.0)
    for i in i_band:
        ck.close(
            f"E tangential: Ey odd mirror at s={s_node[i] / H:+.2f}h",
            ey[i, :],
            vector_ratio(s_node[i]) * c,
            1e-12,
        )
        ck.close(
            f"E tangential: Ez odd mirror at s={s_node[i] / H:+.2f}h",
            ez[i, :],
            vector_ratio(s_node[i]) * c,
            1e-12,
        )
    ck.close("E tangential: Ey deep interior zero", ey[i_deep, :], 0.0, 0.0)
    ck.close("E tangential: Ez deep interior zero", ez[i_deep, :], 0.0, 0.0)

    # --- 2) tangential E: linear continuation through the surface ----------
    # Ey = Ez = b*s in the fluid must continue as b*s through the ghosts
    # (the bilinear gather is exact on linear fields): Dirichlet 0 at the wall
    set_xprofile(Ey, b * s_node)
    set_xprofile(Ez, b * s_node)
    Ex[...] = 0.0
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    ey = arr2d(Ey)
    ez = arr2d(Ez)
    for i in i_band:
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

    # --- 4) B: swapped parity (normal Bx odd, cut By/Bz conformal-owned) ---
    # Bx lives on the nodal-x rows, so the odd closed forms match Ey/Ez; the
    # cut By face (Sy = 0.3 != 0) and the cut Bz x-edge (lx = 0.3 != 0) keep
    # eb_update = 1 under the conformal marking and, with
    # fill_covered_centers = false for B, stay solver-owned (untouched)
    Bx[...] = c
    By[...] = c
    Bz[...] = c
    wx.hybrid_apply_eb_boundary_to_face_field("Bfield_fp", 0)
    bx = arr2d(Bx)
    by = arr2d(By)
    bz = arr2d(Bz)
    ck.close("B normal: fluid rows untouched", bx[i_fluid, :], c, 0.0)
    for i in i_band:
        ck.close(
            f"B normal: odd mirror at s={s_node[i] / H:+.2f}h",
            bx[i, :],
            vector_ratio(s_node[i]) * c,
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
            f"B in-plane flux: cut By face left to the conformal update at s={s_cent[i] / H:+.2f}h",
            by[i, :],
            c,
            0.0,
        )
        ck.close(
            f"B tangential: cut Bz x-edge left to the conformal update at s={s_cent[i] / H:+.2f}h",
            bz[i, :],
            c,
            0.0,
        )
    ck.close("B in-plane flux: By deep zero", by[cent_rows(-100.0, -1.0), :], 0.0, 0.0)
    ck.close("B tangential: Bz deep zero", bz[cent_rows(-100.0, -1.0), :], 0.0, 0.0)

    # --- 5) nodal scalars: rho odd (Dirichlet 0), Pe even (Neumann) --------
    rho[...] = c
    wx.hybrid_apply_eb_boundary_to_nodal_scalar("rho_fp", 0, True)
    r = arr2d(rho)
    ck.close("rho odd: fluid untouched", r[i_fluid, :], c, 0.0)
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
    ck.close("Pe even: fluid untouched", p[i_fluid, :], c, 0.0)
    for i in i_band:
        ck.close(
            f"Pe even: Neumann mirror at s={s_node[i] / H:+.2f}h", p[i, :], c, 1e-12
        )
    ck.close("Pe even: deep zero", p[i_deep, :], 0.0, 0.0)

    # --- 6) deposit fold: PEC image parities, planar closed forms ----------
    # The covered node row at s=-0.7h holds a deposit c (the shape-function
    # spill of wall-adjacent particles). Its mirror lands 0.6 of the way
    # between the first two fluid rows, so the fold delivers -0.6c and -0.4c
    # (PEC image charge: subtracted) and conserves the folded amount. Under
    # the conformal masks the s=+0.3h row is S_SOLUTION and receives its
    # share (the RZ staircase masks would skip it).
    i_dep = i_band  # s = -0.7h (single covered row for this wall)
    i_first = node_rows(0.05, 0.5)  # s = +0.3h
    i_second = node_rows(1.0, 1.5)  # s = +1.3h
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
    # x-normal wall): image current antiparallel (subtracted), same geometry
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
    ck.close("fold Jy tangential: subtracted -0.6c", jy[i_first, :], -0.6 * c, 1e-12)
    ck.close("fold Jy tangential: subtracted -0.4c", jy[i_second, :], -0.4 * c, 1e-12)
    ck.close("fold Jz tangential: subtracted -0.6c", jz[i_first, :], -0.6 * c, 1e-12)
    ck.close("fold Jz tangential: subtracted -0.4c", jz[i_second, :], -0.4 * c, 1e-12)

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
        "fold Jy tangential (reflect): added +0.6c", jy[i_first, :], 0.6 * c, 1e-12
    )
    ck.close(
        "fold Jy tangential (reflect): added +0.4c", jy[i_second, :], 0.4 * c, 1e-12
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
        "selectivity: varying field bit-identical in the fluid",
        after[i_fluid, :],
        before[i_fluid, :],
        0.0,
    )

    ck.finish()


# ----------------------------------------------------------------------------
# Transitional Marder battery (planar wall; ports the 3D battery with the 2D
# nodal divergence d(Ex)/dx + d(Ez)/dz and the in-plane-only update)
# ----------------------------------------------------------------------------
def run_marder_battery(sim):
    wx = sim.extension.warpx
    ck = CheckSet()

    q_e = constants.q_e
    N_FLOOR = 1.0e16  # matches the solver n_floor
    RHO_FLOOR = N_FLOOR * q_e
    RHO_DENSE = 1.0e18 * q_e
    RHO_TRANS = 0.5e16 * q_e  # strictly inside (0, RHO_FLOOR]
    P0 = 1.0

    # x-aligned bands by node index: dense | transition | vacuum | wall
    I_DENSE_END = 8
    I_TRANS_END = 16

    rho_prof = np.zeros(N_X + 1)
    rho_prof[:I_DENSE_END] = RHO_DENSE
    rho_prof[I_DENSE_END:I_TRANS_END] = RHO_TRANS
    # Pe falls smoothly to zero at the vacuum boundary (zero slope there)
    pe_prof = np.zeros(N_X + 1)
    i_nodes = np.arange(I_TRANS_END + 1)
    pe_prof[: I_TRANS_END + 1] = P0 * np.cos(np.pi * i_nodes / (2 * I_TRANS_END)) ** 2

    rho = fields.RhoFPWrapper()
    pe = fields.ElectronPressureFPWrapper()
    set_xprofile(rho, rho_prof)
    set_xprofile(pe, pe_prof)

    Ex = fields.ExFPWrapper()
    Ey = fields.EyFPWrapper()
    Ez = fields.EzFPWrapper()
    Bx = fields.BxFPWrapper()
    By = fields.ByFPWrapper()
    Bz = fields.BzFPWrapper()
    Jx = fields.JxFPWrapper()
    Jy = fields.JyFPWrapper()
    Jz = fields.JzFPWrapper()
    for w in (Bx, By, Bz, Jx, Jy, Jz):
        w[...] = 0.0

    # numpy mirrors of the kernel's 2-point stencils and masks --------------
    # x-edge (cell-centered in x) interpolated rho and the closed-form
    # pressure-only target Ex = -grad(Pe)/max(rho, floor); Ey = Ez = 0
    rho_xedge = 0.5 * (rho_prof[:-1] + rho_prof[1:])
    rho_lim_x = np.maximum(rho_xedge, RHO_FLOOR)
    ex_target = -(np.diff(pe_prof) / H) / rho_lim_x

    # update/mask bands
    band_xedge = (rho_xedge > 0.0) & (rho_xedge <= RHO_FLOOR)  # Ex rows
    band_node = (rho_prof > 0.0) & (rho_prof <= RHO_FLOOR)  # nodal (Ez) rows
    dense_xedge = rho_xedge > RHO_FLOOR
    # interior vacuum x-edge rows, clear of the wall fill band and the
    # domain boundary
    vac_rows = [i for i in range(N_X) if rho_xedge[i] <= 0.0 and 17 <= i <= 21]
    dense_rows = [i for i in range(1, N_X) if dense_xedge[i]]
    band_rows = [i for i in range(N_X) if band_xedge[i]]

    def divergence(ex, ez):
        """Nodal divergence with the kernel's downward stencils (2D: no y
        term). Yee global shapes: ex (N, N+1), ez (N+1, N); result is nodal
        (N+1, N+1) with periodic wrap in z and one-sided x boundary nodes
        left at zero."""
        div = np.zeros((N_X + 1, N_Z + 1))
        div[1:-1, :] += (ex[1:, :] - ex[:-1, :]) / H
        dvz = (ez - np.roll(ez, 1, axis=1)) / H  # at j = 0..N-1, j=0 wraps
        div[:, :-1] += dvz
        div[:, -1] += dvz[:, 0]  # duplicated periodic node row
        return div

    def masked_err_norm(div, div_target_1d):
        err = div - div_target_1d[:, None]
        err[~band_node, :] = 0.0
        err[[0, -1], :] = 0.0  # one-sided boundary nodes excluded
        return float(np.sqrt(np.sum(err**2)))

    def set_baseline(noise_amp, seed):
        rng = np.random.RandomState(seed)
        ex0 = np.broadcast_to(ex_target[:, None], arr2d(Ex).shape).copy()
        ez0 = np.zeros(arr2d(Ez).shape)
        scale = noise_amp * np.max(np.abs(ex_target))
        ex0[band_rows, :] += scale * (rng.rand(len(band_rows), *ex0.shape[1:]) - 0.5)
        i_bnode = [i for i in range(1, N_X) if band_node[i]]
        ez0[i_bnode, :] += scale * (rng.rand(len(i_bnode), *ez0.shape[1:]) - 0.5)
        Ex[...] = ex0
        Ey[...] = 0.0
        Ez[...] = ez0

    # the divergence target of the "ohm"/"grad_pe_only" modes with J = B = 0
    div_target = divergence(
        np.broadcast_to(ex_target[:, None], arr2d(Ex).shape),
        np.zeros(arr2d(Ez).shape),
    )[:, 0]

    # --- 1) cleans divergence error (Phoenix test_cleans_divergence_error) -
    set_baseline(0.3, seed=42)
    err_before = masked_err_norm(divergence(arr2d(Ex), arr2d(Ez)), div_target)
    n_iter, resid = wx.hybrid_marder_correct_e(
        0, alpha=0.1, max_iterations=500, rtol=1e-2
    )
    err_after = masked_err_norm(divergence(arr2d(Ex), arr2d(Ez)), div_target)
    ck.expect(
        "marder: converged before the iteration cap",
        n_iter < 500,
        f"n_iter={n_iter}",
    )
    ck.expect(
        "marder: cleans the divergence error in the band",
        err_after < 0.15 * err_before,
        f"before={err_before:.3e} after={err_after:.3e}",
    )

    # --- 2+3) modifies the transition band only; vacuum untouched; the
    # out-of-plane Ey is bit-identical through the call (the 2D update
    # touches Ex/Ez only). Ey is pre-settled: zero on the PEC x boundary
    # rows and EB-filled once, so the in-call boundary applications rewrite
    # values identical to those already present.
    Ex[...] = 1.0e5
    Ez[...] = 0.0
    shape_ey = np.asarray(Ey[...]).shape
    idx = np.indices(shape_ey)
    ey0 = 2.0e5 * (1.0 + 0.05 * idx[0] + 0.013 * idx[1])
    ey0[0, ...] = 0.0
    ey0[-1, ...] = 0.0
    Ey[...] = ey0
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    before_ex = np.array(arr2d(Ex))
    before_ey = np.array(np.asarray(Ey[...]))
    wx.hybrid_marder_correct_e(0, alpha=0.1, max_iterations=50, rtol=0.0, atol=0.0)
    ex = arr2d(Ex)
    ck.close(
        "marder: dense plasma core bit-identical",
        ex[dense_rows, :],
        before_ex[dense_rows, :],
        0.0,
    )
    d_band = np.max(np.abs(ex[band_rows, :] - before_ex[band_rows, :]))
    ck.expect(
        "marder: transition band is modified", d_band > 0.0, f"max|dE|={d_band:.3e}"
    )
    ck.close(
        "marder: vacuum gap bit-identical",
        ex[vac_rows, :],
        before_ex[vac_rows, :],
        0.0,
    )
    ck.close(
        "marder: out-of-plane Ey bit-identical through the 2D update",
        np.asarray(Ey[...]),
        before_ey,
        0.0,
    )

    # --- 4) monotone convergence (single-iteration calls; the returned
    # residual is measured before each update, Phoenix semantics) -----------
    set_baseline(0.5, seed=123)
    residuals = []
    for _ in range(20):
        _, resid = wx.hybrid_marder_correct_e(
            0, alpha=0.1, max_iterations=1, rtol=0.0, atol=0.0
        )
        residuals.append(resid)
    residuals = np.array(residuals)
    n_increase = int(np.sum(residuals[1:] > residuals[:-1] * (1.0 + 1.0e-10)))
    ck.expect(
        "marder: residual decreases monotonically",
        n_increase <= 2,
        f"increases={n_increase} residuals[0]={residuals[0]:.3e} "
        f"residuals[-1]={residuals[-1]:.3e}",
    )
    ck.expect(
        "marder: residual at least halved over 20 sweeps",
        residuals[-1] < 0.5 * residuals[0],
        f"ratio={residuals[-1] / residuals[0]:.3f}",
    )

    # --- 5) grad_pe_only: identical to ohm when J = B = 0, different once
    # a JxB drive exists ------------------------------------------------------
    set_baseline(0.3, seed=7)
    e0 = [np.array(np.asarray(w[...])) for w in (Ex, Ey, Ez)]
    wx.hybrid_marder_correct_e(0, alpha=0.1, max_iterations=5, rtol=0.0, target="ohm")
    e_ohm = [np.array(np.asarray(w[...])) for w in (Ex, Ey, Ez)]
    for w, e in zip((Ex, Ey, Ez), e0):
        w[...] = e
    wx.hybrid_marder_correct_e(
        0, alpha=0.1, max_iterations=5, rtol=0.0, target="grad_pe_only"
    )
    e_gpe = [np.array(np.asarray(w[...])) for w in (Ex, Ey, Ez)]
    # The two calls start from identical valid data but inherit different
    # deep-ghost history (the restore rewrites valid data only, and the
    # iteration fills exchange a single ghost layer), which shows up at the
    # few-percent level on GPU arenas: compare at a tolerance two orders
    # below the genuine JxB-drive signal (~14 in these units) instead of
    # bitwise.
    e_scale = max(float(np.max(np.abs(e))) for e in e0)
    for name, a, b2 in zip("xyz", e_ohm, e_gpe):
        ck.close(
            f"marder: grad_pe_only == ohm for J=B=0 (E{name})",
            a / e_scale,
            np.asarray(b2) / e_scale,
            5e-2,
        )

    # turn on a uniform JxB drive: enE_x = -Jy*Bz / rho(x) couples to the
    # dense edge of the band, so the two targets must now differ
    Jy[...] = 1.0e3
    Bz[...] = 0.1
    for w, e in zip((Ex, Ey, Ez), e0):
        w[...] = e
    wx.hybrid_marder_correct_e(0, alpha=0.1, max_iterations=5, rtol=0.0, target="ohm")
    ex_ohm2 = np.array(arr2d(Ex))
    for w, e in zip((Ex, Ey, Ez), e0):
        w[...] = e
    wx.hybrid_marder_correct_e(
        0, alpha=0.1, max_iterations=5, rtol=0.0, target="grad_pe_only"
    )
    ex_gpe2 = arr2d(Ex)
    d_t = np.max(np.abs(ex_ohm2[band_rows, :] - ex_gpe2[band_rows, :]))
    ck.expect(
        "marder: ohm and grad_pe_only differ under a JxB drive",
        d_t > 0.0,
        f"max|dE|={d_t:.3e}",
    )

    # --- 6) Marder RE-APPLIES the EB boundary condition ---------------------
    # The transitional correction must leave E satisfying the EB BC (the
    # in-loop ApplyPECBoundaryToField, MarderCorrection.cpp:543, runs after
    # every E update). Seed a consistent baseline, make it EB-consistent, then
    # deliberately CORRUPT the covered (x > X_WALL) region -- a BC violation.
    # The covered cells are eb_update_E = 0, so the div-correction never touches
    # them (and the transition band at rows 8-16 is far from the wall node 24,
    # so the corruption does not perturb the correction either): the covered
    # values are set ONLY by the in-loop EB re-apply. After Marder the field
    # must again satisfy the EB fill, i.e. re-applying the EB operator is a
    # no-op (idempotent), and the injected garbage is gone.
    Jy[...] = 0.0
    Bz[...] = 0.0
    set_baseline(0.3, seed=99)
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)  # consistent start
    # covered rows just past the wall node (nodal i >= 25; x-edge i >= 24);
    # element [0] of each is the actual fill-band row (s = -0.7h / -0.2h)
    cov_nodes = list(range(X_WALL_NODE + 1, N_X + 1))
    cov_edges = list(range(X_WALL_NODE, N_X))
    GARBAGE = 1.0e5
    for w, rows in ((Ex, cov_edges), (Ey, cov_nodes), (Ez, cov_nodes)):
        set_xrows(w, rows, GARBAGE)
    wx.hybrid_marder_correct_e(0, alpha=0.1, max_iterations=10, rtol=0.0, atol=0.0)
    post = [np.array(np.asarray(w[...])) for w in (Ex, Ey, Ez)]
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    after_bc = [np.array(np.asarray(w[...])) for w in (Ex, Ey, Ez)]
    e_scale = max(float(np.max(np.abs(p))) for p in post) + 1e-300
    for name, p, q in zip("xyz", post, after_bc):
        ck.close(
            f"marder: post-Marder E satisfies the EB BC (E{name})",
            np.asarray(p) / e_scale,
            np.asarray(q) / e_scale,
            1e-10,
        )
    # and the corrupted fill-band row was actually overwritten (so the
    # idempotency above is a genuine re-apply, not an empty covered region)
    moved = min(
        float(np.min(np.abs(np.asarray(w[...])[rows[0], ...] - GARBAGE)))
        for w, rows in ((Ex, cov_edges), (Ey, cov_nodes), (Ez, cov_nodes))
    )
    ck.expect(
        "marder: corrupted covered fill band was overwritten by the EB re-apply",
        moved > 0.5 * GARBAGE,
        f"min|E_fillband - garbage| = {moved:.3e} (garbage={GARBAGE:.1e})",
    )

    ck.finish()


# ----------------------------------------------------------------------------
# Hyper-resistivity stencil battery (2D XZ): validates the isotropized
# Mehrstellen 9-point Laplacian against its closed form and demonstrates the
# isotropization of its damping rate via plane-wave eigenvalues (exact,
# quadrature-free). Solver configured so the Faraday E solve reduces to
# E = -eta_h * nabla^2(J_plasma).
# ----------------------------------------------------------------------------
def run_hyper_battery(sim):
    wx = sim.extension.warpx
    ck = CheckSet()
    ETA_H = 1.0

    rho = fields.RhoFPWrapper()
    rho[...] = np.full(np.asarray(rho[...]).shape, 100.0 * 1.0e16 * constants.q_e)
    for w in (
        fields.BxFPWrapper(),
        fields.ByFPWrapper(),
        fields.BzFPWrapper(),
        fields.JxFPWrapper(),
        fields.JyFPWrapper(),
        fields.JzFPWrapper(),
        fields.JxFPPlasmaWrapper(),
        fields.JyFPPlasmaWrapper(),
        fields.JzFPPlasmaWrapper(),
    ):
        w[...] = 0.0

    # Ey/Jy are the out-of-plane (nodal x nodal z) pair: their laplacian is
    # purely in-plane, the cleanest probe of the 9-point stencil
    Jpy = fields.JyFPPlasmaWrapper()
    Ey = fields.EyFPWrapper()

    def d2(a, ax):
        return np.roll(a, -1, ax) - 2.0 * a + np.roll(a, 1, ax)

    def lap_cross(a):
        return (d2(a, 0) + d2(a, 1)) / H**2

    def lap_iso(a):
        return lap_cross(a) + d2(d2(a, 0), 1) / (6.0 * H**2)

    shape = np.asarray(Jpy[...]).shape
    xn = LO + np.arange(shape[0]) * H
    zn = LO + np.arange(shape[1]) * H
    X, Z = np.meshgrid(xn, zn, indexing="ij")
    interior = (slice(4, 21), slice(2, 31))

    # --- 1) stencil exactness on a random field ---------------------------
    rng = np.random.RandomState(2025)
    jy = rng.rand(*shape) - 0.5
    Jpy[...] = jy
    wx.hybrid_solve_e(True)
    ey = np.asarray(Ey[...])
    expected = -ETA_H * lap_iso(jy)
    scale = float(np.max(np.abs(expected[interior])))
    ck.close(
        "hyper 2D: solver matches the 9-point Mehrstellen closed form",
        ey[interior] / scale,
        expected[interior] / scale,
        1e-12,
    )

    # --- 2) consistency: exact on quadratics ------------------------------
    Jpy[...] = X**2
    wx.hybrid_solve_e(True)
    ey = np.asarray(Ey[...])
    ck.close(
        "hyper 2D: exact on quadratics (laplacian of x^2 is 2)",
        ey[interior],
        -ETA_H * 2.0,
        1e-9,
    )

    # --- 3) isotropization: plane-wave damping rates ----------------------
    kmag = 1.0 / H  # kh = 1

    def solver_rate(j_field):
        Jpy[...] = j_field
        wx.hybrid_solve_e(True)
        lam = (np.asarray(Ey[...]) / (ETA_H * j_field))[interior]
        return float(np.median(lam[np.abs(j_field[interior]) > 0.5]))

    def stencil_rate(lap, j_field):
        lam = (-lap(j_field) / j_field)[interior]
        return float(np.median(lam[np.abs(j_field[interior]) > 0.5]))

    j_axis = np.cos(kmag * X)
    j_diag = np.cos(kmag * (X + Z) / np.sqrt(2.0))
    aniso_iso = abs(solver_rate(j_axis) - solver_rate(j_diag)) / kmag**2
    aniso_cross = (
        abs(stencil_rate(lap_cross, j_axis) - stencil_rate(lap_cross, j_diag)) / kmag**2
    )
    ratio = aniso_iso / max(aniso_cross, 1e-300)
    ck.expect(
        "hyper 2D: axis/diagonal damping anisotropy suppressed (>= 4x at kh=1)",
        ratio < 0.25,
        f"aniso(iso)/aniso(cross) = {ratio:.4f} "
        f"(cross = {aniso_cross:.4f} k^2, iso = {aniso_iso:.5f} k^2)",
    )

    ck.finish()


# ----------------------------------------------------------------------------
# Resistive isotropization battery (2D XZ): the corner-curl correction makes
# the Faraday curl emit the in-plane (xz) Mehrstellen Laplacian of the
# out-of-plane B (By), cancelling the cross-stencil cos(4*theta) anisotropy.
# Validated by isolating the correction (ion current zero): the Faraday curl
# of E reproduces -(eta/mu0)*corner(By) to round-off, the emergent operator
# is isotropic, and the correction vanishes on quadratics.
# ----------------------------------------------------------------------------
def run_resistive_battery(sim):
    wx = sim.extension.warpx
    ck = CheckSet()
    mu0 = constants.mu0
    ETA_R = 1.0
    h = H

    rho = fields.RhoFPWrapper()
    rho[...] = np.full(np.asarray(rho[...]).shape, 100.0 * 1.0e16 * constants.q_e)
    for w in (
        fields.JxFPWrapper(),
        fields.JyFPWrapper(),
        fields.JzFPWrapper(),
        fields.JxFPPlasmaWrapper(),
        fields.JyFPPlasmaWrapper(),
        fields.JzFPPlasmaWrapper(),
        fields.BxFPWrapper(),
        fields.BzFPWrapper(),
        fields.ExFPWrapper(),
        fields.EyFPWrapper(),
        fields.EzFPWrapper(),
    ):
        w[...] = 0.0

    By = fields.ByFPWrapper()
    bshp = np.asarray(By[...]).shape
    NC = (N_X, N_Z)

    def crop(a):
        return np.asarray(a)[: NC[0], : NC[1]]

    def d2(a, ax):
        return np.roll(a, -1, ax) - 2.0 * a + np.roll(a, 1, ax)

    def upx(F):
        return (np.roll(F, -1, 0) - F) / h

    def upz(F):
        return (np.roll(F, -1, 1) - F) / h

    ii = (slice(4, 22), slice(4, 28))  # stay left of the x-wall (node 24.3)

    # --- 1) the correction's Faraday curl is the in-plane (xz) corner -------
    rng = np.random.RandomState(11)
    by = rng.rand(NC[0], NC[1]) - 0.5
    By[...] = np.broadcast_to(
        by.reshape(NC[0], NC[1])[: bshp[0], : bshp[1]], bshp
    ).copy()
    wx.hybrid_solve_e(True)
    Ex = crop(fields.ExFPWrapper()[...])
    Ez = crop(fields.EzFPWrapper()[...])
    corner = (1.0 / (6.0 * h * h)) * d2(d2(by, 0), 1)
    curly = upz(Ex) - upx(Ez)  # curl_up(E)_y
    expected = -(ETA_R / mu0) * corner
    scale = float(np.max(np.abs(expected[ii])))
    ck.close(
        "resistive 2D: Faraday curl of the correction is the in-plane corner",
        curly[ii] / scale,
        expected[ii] / scale,
        1e-12,
    )

    # --- 2) emergent operator isotropic; teeth ------------------------------
    lap_cross = (d2(by, 0) + d2(by, 1)) / h**2
    lap_iso = lap_cross + corner
    emergent = lap_cross - (mu0 / ETA_R) * curly
    ck.close(
        "resistive 2D: emergent operator equals the isotropic Laplacian",
        emergent[ii],
        lap_iso[ii],
        1e-9 * float(np.max(np.abs(lap_iso[ii]))),
    )

    # --- 3) plane-wave anisotropy suppressed --------------------------------
    xc = (np.arange(NC[0]) + 0.5) * h + LO
    zc = (np.arange(NC[1]) + 0.5) * h + LO
    X, Z = np.meshgrid(xc, zc, indexing="ij")
    kmag = 1.0 / h

    def aniso(lap_fn):
        ja = np.cos(kmag * X)
        jd = np.cos(kmag * (X + Z) / np.sqrt(2.0))
        la = (-lap_fn(ja) / ja)[ii]
        ld = (-lap_fn(jd) / jd)[ii]
        return abs(float(np.median(la)) - float(np.median(ld))) / kmag**2

    a_cross = aniso(lambda a: (d2(a, 0) + d2(a, 1)) / h**2)
    a_iso = aniso(
        lambda a: (d2(a, 0) + d2(a, 1)) / h**2 + (1.0 / (6.0 * h * h)) * d2(d2(a, 0), 1)
    )
    ck.expect(
        "resistive 2D: axis/diagonal anisotropy suppressed (>= 4x at kh=1)",
        a_iso < 0.25 * a_cross,
        f"aniso(iso)/aniso(cross)={a_iso / max(a_cross, 1e-300):.4f}",
    )

    # --- 4) pure truncation canceller ---------------------------------------
    By[...] = np.broadcast_to((X**2)[: bshp[0], : bshp[1]], bshp).copy()
    wx.hybrid_solve_e(True)
    ex = crop(fields.ExFPWrapper()[...])
    ck.close(
        "resistive 2D: correction vanishes on quadratics",
        ex[ii],
        0.0,
        1e-9 * float(np.max(np.abs(ex))) + 1e-12,
    )

    ck.finish()


# ----------------------------------------------------------------------------
# Div(B) Marder-clean battery (collocated planar wall)
# ----------------------------------------------------------------------------
def run_divb_clean_battery(sim):
    """Div(B) Marder clean on a collocated grid: the band-restricted
    ``B += alpha*grad(div B)`` diffusion that dissipates the curved-wall
    divergence the EB mirror fill injects on the conformal-EB collocated B
    push. The correction is a pure gradient, so curl(B) -- and hence the
    plasma current J = curl(B) it feeds -- is preserved to round-off while
    only the near-wall band divergence is reduced. Exercises the production
    parity (magnetic: normal odd) through the hybrid_marder_clean_divergence
    binding.

    Key invariant tested: for the centered nodal grad the kernel uses, a
    centered nodal curl of ``grad(scalar)`` vanishes identically, so curl(B)
    is unchanged to round-off in the band interior regardless of how the
    divergence itself is computed.
    """
    wx = sim.extension.warpx
    ck = CheckSet()

    L = HI - LO
    nx, nz = N_X + 1, N_Z + 1  # collocated: every field is nodal
    x_node = LO + np.arange(nx) * H
    z_node = LO + np.arange(nz) * H
    s_cell = (X_WALL - x_node) / H  # signed distance in cells (fluid s > 0)

    CLEAN_CELLS = 6.0  # the clean band (grad(div) acts on |phi| <= 6 h)
    FILL_CELLS = 1.0  # the EB mirror re-fill band (rewrites |phi| <~ 1 h)

    Bx = fields.BxFPWrapper()
    By = fields.ByFPWrapper()
    Bz = fields.BzFPWrapper()
    for w in (Bx, By, Bz):
        w[...] = 0.0

    # Analytic seed (all nodal):
    #   Bx = A sin(2 pi (z-LO)/L) + P(x) ; Bz = 0 ; By = 0
    #   curl_y = dBx/dz - dBz/dx = A (2 pi/L) cos(...)   (z-periodic, P drops out)
    #   div    = dBx/dx + dBz/dz = dP/dx                 (localized in the band)
    # P(x): a smooth fluid-side bump peaked ~2.5 cells inside the wall, zero in
    # the bulk and inside the conductor, so the seeded divergence lives in the
    # near-wall band -- mimicking the curved-wall injection of the mirror fill.
    A = 3.0
    sinz = np.sin(2.0 * np.pi * (z_node - LO) / L)
    P = 0.7 * np.exp(-((s_cell - 2.5) ** 2) / (2.0 * 1.5**2))
    P[s_cell <= 0.0] = 0.0
    bx0 = np.broadcast_to(A * sinz[None, :] + P[:, None], arr2d(Bx).shape).copy()
    bz0 = np.zeros(arr2d(Bz).shape)
    Bx[...] = bx0
    Bz[...] = bz0
    By[...] = 0.0

    def div_nodal(bx, bz):
        """Centered nodal divergence dBx/dx + dBz/dz (z periodic, x interior)."""
        d = np.zeros_like(bx)
        d[1:-1, :] += (bx[2:, :] - bx[:-2, :]) / (2.0 * H)
        d += (np.roll(bz, -1, axis=1) - np.roll(bz, 1, axis=1)) / (2.0 * H)
        return d

    def curl_y_nodal(bx, bz):
        """Centered nodal curl_y = dBx/dz - dBz/dx (z periodic, x interior)."""
        c = (np.roll(bx, -1, axis=1) - np.roll(bx, 1, axis=1)) / (2.0 * H)
        c[1:-1, :] -= (bz[2:, :] - bz[:-2, :]) / (2.0 * H)
        return c

    bx_before = np.array(arr2d(Bx))
    bz_before = np.array(arr2d(Bz))
    div_before = div_nodal(bx_before, bz_before)
    curl_before = curl_y_nodal(bx_before, bz_before)

    # band-interior window: fluid nodes well clear of both the fill re-write
    # band and the outer clean-band edge, so the centered stencils' x-neighbors
    # are also inside the actively-corrected band.
    interior = (s_cell >= FILL_CELLS + 1.0) & (s_cell <= CLEAN_CELLS - 1.0)
    # bulk fluid the clean must not touch -- a window safely past the clean
    # band's smooth tail and clear of the x-domain (Dirichlet) boundary nodes.
    deep = (s_cell >= CLEAN_CELLS + 4.0) & (s_cell <= 20.0)

    # Run a single sweep of the production div(B) clean (magnetic parity, B
    # mask, nullptr cache). One sweep is the exact unit of the algorithm: the
    # centered nodal grad(div) is applied once in the band, so curl(B) is
    # preserved to round-off. (The production cadence takes a few such sweeps;
    # on this idealized planar wall many sweeps would slowly pump the centered
    # stencil's undamped checkerboard null-mode, which the per-sweep invariant
    # below is the clean test of.)
    wx.hybrid_marder_clean_divergence(
        field="Bfield_fp",
        lev=0,
        alpha=0.1,
        max_iters=1,
        clean_band_cells=CLEAN_CELLS,
        fill_band_cells=FILL_CELLS,
    )

    bx_after = np.array(arr2d(Bx))
    bz_after = np.array(arr2d(Bz))
    div_after = div_nodal(bx_after, bz_after)
    curl_after = curl_y_nodal(bx_after, bz_after)

    # --- 1) the clean actually modified the band ---------------------------
    d_band = float(np.max(np.abs(bx_after[interior, :] - bx_before[interior, :])))
    ck.expect(
        "divb_clean: the band B field is modified",
        d_band > 0.0,
        f"max|dBx|={d_band:.3e}",
    )

    # --- 2) the divergence in the band is reduced --------------------------
    dn_before = float(np.sqrt(np.sum(div_before[interior, :] ** 2)))
    dn_after = float(np.sqrt(np.sum(div_after[interior, :] ** 2)))
    ck.expect(
        "divb_clean: the band div(B) is reduced",
        dn_after < 0.9 * dn_before,
        f"before={dn_before:.3e} after={dn_after:.3e}",
    )

    # --- 3) curl(B) is preserved to round-off (pure-gradient correction) ---
    # This is the safety guarantee of the clean: a centered nodal curl of the
    # centered nodal grad(div) vanishes identically, so J = curl(B) is left
    # untouched while the divergence is dissipated.
    curl_scale = float(np.max(np.abs(curl_before[interior, :])))
    dcurl = float(np.max(np.abs(curl_after[interior, :] - curl_before[interior, :])))
    ck.expect(
        "divb_clean: curl(B) preserved to round-off in the band",
        dcurl <= 1.0e-12 * curl_scale,
        f"max|dcurl|={dcurl:.3e} curl_scale={curl_scale:.3e}",
    )

    # --- 4) the bulk fluid is left essentially untouched (band-restricted) --
    deep_chg = float(np.max(np.abs(bx_after[deep, :] - bx_before[deep, :])))
    ck.expect(
        "divb_clean: deep fluid is unchanged (band-restricted)",
        deep_chg < 1.0e-3,
        f"max|dBx_deep|={deep_chg:.3e}",
    )

    # --- 5) the same clean on the total Ampere current (div(J), electric
    # parity / E mask): the kernel is identical, so curl(J) -- the physical
    # current the solver feeds back -- is likewise preserved to round-off while
    # the band divergence is reduced. Exercises the binding's
    # hybrid_current_fp_plasma branch.
    Jx = fields.JxFPPlasmaWrapper()
    Jy = fields.JyFPPlasmaWrapper()
    Jz = fields.JzFPPlasmaWrapper()
    for w in (Jx, Jy, Jz):
        w[...] = 0.0
    Jx[...] = np.broadcast_to(A * sinz[None, :] + P[:, None], arr2d(Jx).shape).copy()
    Jz[...] = np.zeros(arr2d(Jz).shape)
    jx0, jz0 = np.array(arr2d(Jx)), np.array(arr2d(Jz))
    jdiv0 = div_nodal(jx0, jz0)
    jcurl0 = curl_y_nodal(jx0, jz0)
    wx.hybrid_marder_clean_divergence(
        field="hybrid_current_fp_plasma",
        lev=0,
        alpha=0.1,
        max_iters=1,
        clean_band_cells=CLEAN_CELLS,
        fill_band_cells=FILL_CELLS,
    )
    jx1, jz1 = np.array(arr2d(Jx)), np.array(arr2d(Jz))
    jdiv1 = div_nodal(jx1, jz1)
    jcurl1 = curl_y_nodal(jx1, jz1)
    jdn0 = float(np.sqrt(np.sum(jdiv0[interior, :] ** 2)))
    jdn1 = float(np.sqrt(np.sum(jdiv1[interior, :] ** 2)))
    ck.expect(
        "divj_clean: the band div(J) is reduced",
        jdn1 < 0.9 * jdn0,
        f"before={jdn0:.3e} after={jdn1:.3e}",
    )
    jcurl_scale = float(np.max(np.abs(jcurl0[interior, :])))
    jdcurl = float(np.max(np.abs(jcurl1[interior, :] - jcurl0[interior, :])))
    ck.expect(
        "divj_clean: curl(J) preserved to round-off in the band",
        jdcurl <= 1.0e-12 * jcurl_scale,
        f"max|dcurl|={jdcurl:.3e} curl_scale={jcurl_scale:.3e}",
    )

    ck.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--battery",
        choices=["eb", "marder", "hyper", "resistive", "divb_clean"],
        default="eb",
        help="which unit battery to run (eb fills/folds, the transitional "
        "Marder correction, the isotropized hyper-resistivity stencil, or the "
        "collocated div(B) Marder clean)",
    )
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    sim = setup_simulation(
        hyper=(args.battery == "hyper"),
        resistive=(args.battery == "resistive"),
        collocated=(args.battery == "divb_clean"),
    )
    if args.battery == "resistive":
        run_resistive_battery(sim)
    elif args.battery == "hyper":
        run_hyper_battery(sim)
    elif args.battery == "marder":
        run_marder_battery(sim)
    elif args.battery == "divb_clean":
        run_divb_clean_battery(sim)
    else:
        run_plane_battery(sim)


if __name__ == "__main__":
    main()
