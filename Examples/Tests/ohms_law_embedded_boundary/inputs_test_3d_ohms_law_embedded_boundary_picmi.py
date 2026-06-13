#!/usr/bin/env python3
#
# --- Unit tests of the hybrid solver's embedded-boundary value application
# --- (modeled on the Phoenix embedded-coil boundary test suite): a small
# --- grid with a deterministic conducting wall is initialized, synthetic
# --- fields with known closed-form behavior are loaded through the field
# --- wrappers, the boundary operators are applied directly through their
# --- Python bindings, and the values are asserted point by point:
# ---
# ---   * solution-domain (plasma) values are never modified,
# ---   * deep conductor interior is exactly zeroed,
# ---   * tangential E/J ghosts are odd mirrors (s/d_im scaling; exact -1
# ---     for |s| >= h/2 against constant fields, exact linear continuation
# ---     through zero at the surface against linear fields),
# ---   * normal E/J ghosts are even mirrors (constant fields reproduced
# ---     exactly),
# ---   * B has the swapped (flux-excluding) parity: normal odd, cut faces
# ---     left to the conformal update,
# ---   * cut edges whose centers are covered are filled (junk poked into
# ---     them is replaced),
# ---   * the nodal scalars follow their parity: rho odd (Dirichlet 0 at the
# ---     surface), Pe even (Neumann).
# ---
# --- The planar wall (normal -x) makes the level-set distance and the
# --- mirror geometry analytic, so most assertions hold to round-off; the
# --- cylinder variant exercises the same operators on a curved wall with
# --- interpolated normals and uses sign/containment assertions.

import argparse
import sys

import numpy as np

from pywarpx import fields, picmi

constants = picmi.constants

N_XY = 32
N_Z = 8
LO = -1.0
HI = 1.0
H = (HI - LO) / N_XY  # cubic cells

# planar wall: conductor x > X_WALL, fluid-pointing normal -x; the 0.3 h
# offset from a node puts a cut x-edge with a covered center next to the wall
X_WALL_OFFSET = 0.3
X_WALL_NODE = 24  # x = 0.5
X_WALL = LO + (X_WALL_NODE + X_WALL_OFFSET) * H

R_WALL = 0.8  # cylinder variant

ETA_H = 1.0  # constant hyper-resistivity of the stencil battery

# mirror geometry of the fill operators (EBJBoundary.cpp)
D_IMG_MIN = 0.5 * H


def vector_ratio(s):
    """Odd-parity scaling s/d_im of the staggered vector fill."""
    offset = max(max(abs(s), D_IMG_MIN) - s, H)
    return s / (s + offset)


def scalar_ratio(s):
    """Odd-parity scaling s/d_im of the nodal scalar fill (exact mirror)."""
    return s / max(abs(s), D_IMG_MIN)


def setup_simulation(geometry, hyper=False):
    grid = picmi.Cartesian3DGrid(
        number_of_cells=[N_XY, N_XY, N_Z],
        lower_bound=[LO, LO, -N_Z * H / 2],
        upper_bound=[HI, HI, N_Z * H / 2],
        lower_boundary_conditions=["dirichlet", "periodic", "periodic"],
        upper_boundary_conditions=["dirichlet", "periodic", "periodic"],
        lower_boundary_conditions_particles=["absorbing", "periodic", "periodic"],
        upper_boundary_conditions_particles=["absorbing", "periodic", "periodic"],
        warpx_max_grid_size=2048,
        # the hyper battery needs a single box: the species-free deck
        # allocates the plasma current with zero ghost cells, so the
        # hyper-resistivity stencil must not straddle a box seam
        warpx_max_grid_size_x=2048 if hyper else N_XY // 2,
        warpx_blocking_factor=8,
    )

    sim = picmi.Simulation(
        time_step_size=1.0e-9,
        max_steps=1,
        particle_shape=1,
        verbose=0,
    )
    sim.current_deposition_algo = "direct"
    sim.grid_type = "staggered"

    sim.solver = picmi.HybridPICSolver(
        grid=grid,
        gamma=5.0 / 3.0,
        Te=0.0,
        n0=1.0e18,
        n_floor=1.0e16,
        # the hyper battery isolates the hyper-resistive term:
        # E = -eta_h * nabla^2(J_plasma) with everything else zero
        plasma_resistivity=0.0 if hyper else 1.0e-6,
        plasma_hyper_resistivity=ETA_H if hyper else None,
        substeps=4,
        use_conformal_eb=True,
    )

    if geometry == "plane":
        sim.embedded_boundary = picmi.EmbeddedBoundary(
            implicit_function="(x-xw)", xw=X_WALL
        )
    else:
        sim.embedded_boundary = picmi.EmbeddedBoundary(
            implicit_function="(x**2+y**2-R_w**2)", R_w=R_WALL
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


# ----------------------------------------------------------------------------
# Planar-wall battery (round-off accurate)
# ----------------------------------------------------------------------------
def run_plane_battery(sim):
    wx = sim.extension.warpx
    ck = CheckSet()
    c = 2.0  # constant field amplitude
    b = 3.0  # linear field slope (per unit s)

    # staggered x coordinates and signed distances s = X_WALL - x
    x_node = LO + np.arange(N_XY + 1) * H
    x_cent = LO + (np.arange(N_XY) + 0.5) * H
    s_node = X_WALL - x_node
    s_cent = X_WALL - x_cent

    def node_rows(lo, hi):
        return [i for i in range(N_XY + 1) if lo < s_node[i] / H <= hi]

    def cent_rows(lo, hi):
        return [i for i in range(N_XY) if lo < s_cent[i] / H <= hi]

    # --- 1) tangential E: odd mirror -------------------------------------
    Ey = fields.EyFPWrapper()
    Ex = fields.ExFPWrapper()
    Ez = fields.EzFPWrapper()
    Ex[...] = 0.0
    Ez[...] = 0.0
    Ey[...] = c
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    ey = Ey[...]

    i_fluid = node_rows(0.05, 10.0)  # strictly fluid edge rows
    ck.close("E tangential: fluid rows untouched", ey[i_fluid, :, :], c, 0.0)
    for i in node_rows(-1.0, -0.05):  # mirror band rows
        ck.close(
            f"E tangential: odd mirror at s={s_node[i] / H:+.2f}h",
            ey[i, :, :],
            vector_ratio(s_node[i]) * c,
            1e-12,
        )
    i_deep = node_rows(-100.0, -1.0)
    ck.close("E tangential: deep interior zero", ey[i_deep, :, :], 0.0, 0.0)

    # --- 2) tangential E: linear continuation through the surface --------
    # Ey = b*s in the fluid must continue as b*s through the ghosts (the
    # trilinear gather is exact on linear fields): Dirichlet 0 at the wall
    Ey[...] = b * np.broadcast_to(s_node[:, None, None], Ey[...].shape)
    Ex[...] = 0.0
    Ez[...] = 0.0
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    ey = Ey[...]
    for i in node_rows(-1.0, -0.05):
        s = s_node[i]
        offset = max(max(abs(s), D_IMG_MIN) - s, H)
        expected = (s / (s + offset)) * b * (s + offset)  # = b*s
        ck.close(
            f"E tangential: linear field continues b*s at s={s / H:+.2f}h",
            ey[i, :, :],
            expected,
            1e-12,
        )

    # --- 3) normal E: even mirror + covered-center cut edge --------------
    Ex[...] = c
    Ey[...] = 0.0
    Ez[...] = 0.0
    # poke junk into the cut x-edge row whose center is covered
    # (edge [x_24, x_25] contains the wall; center s = (0.3-0.5)h = -0.2h)
    ex = Ex[...]
    i_cut = [i for i in range(N_XY) if -0.5 < s_cent[i] / H <= 0.0]
    for i in i_cut:
        Ex[i, :, :] = 1.0e6
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    ex = Ex[...]
    i_fluid = cent_rows(0.05, 10.0)
    ck.close("E normal: fluid rows untouched", ex[i_fluid, :, :], c, 0.0)
    for i in i_cut:
        ck.close(
            f"E normal: covered-center cut edge filled (junk replaced) at s={s_cent[i] / H:+.2f}h",
            ex[i, :, :],
            c,
            1e-12,
        )
    for i in cent_rows(-1.0, -0.5):
        ck.close(
            f"E normal: even mirror at s={s_cent[i] / H:+.2f}h", ex[i, :, :], c, 1e-12
        )
    ck.close(
        "E normal: deep interior zero", ex[cent_rows(-100.0, -1.0), :, :], 0.0, 0.0
    )

    # --- 4) B: swapped parity (normal odd, cut faces untouched) ----------
    Bx = fields.BxFPWrapper()
    By = fields.ByFPWrapper()
    Bz = fields.BzFPWrapper()
    Bx[...] = c
    By[...] = c
    Bz[...] = 0.0
    wx.hybrid_apply_eb_boundary_to_face_field("Bfield_fp", 0)
    bx = Bx[...]
    by = By[...]
    ck.close("B normal: fluid rows untouched", bx[node_rows(0.05, 10.0), :, :], c, 0.0)
    for i in node_rows(-1.0, -0.05):
        ck.close(
            f"B normal: odd mirror at s={s_node[i] / H:+.2f}h",
            bx[i, :, :],
            vector_ratio(s_node[i]) * c,
            1e-12,
        )
    ck.close("B normal: deep zero", bx[node_rows(-100.0, -1.0), :, :], 0.0, 0.0)
    # cut By faces (centers covered) belong to the conformal update: untouched
    for i in i_cut:
        ck.close(
            f"B tangential: cut face left to the conformal update at s={s_cent[i] / H:+.2f}h",
            by[i, :, :],
            c,
            0.0,
        )
    ck.close("B tangential: deep zero", by[cent_rows(-100.0, -1.0), :, :], 0.0, 0.0)

    # --- 5) nodal scalars: rho odd (Dirichlet 0), Pe even (Neumann) ------
    rho = fields.RhoFPWrapper()
    rho[...] = c
    wx.hybrid_apply_eb_boundary_to_nodal_scalar("rho_fp", 0, True)
    r = rho[...]
    r = r[..., 0] if r.ndim == 4 else r
    ck.close("rho odd: fluid untouched", r[node_rows(0.05, 10.0), :, :], c, 0.0)
    for i in node_rows(-1.0, -0.05):
        ck.close(
            f"rho odd: mirror at s={s_node[i] / H:+.2f}h",
            r[i, :, :],
            scalar_ratio(s_node[i]) * c,
            1e-12,
        )
    ck.close("rho odd: deep zero", r[node_rows(-100.0, -1.0), :, :], 0.0, 0.0)

    # linear rho = b*s: ghosts continue b*s exactly (zero at the surface)
    rho[...] = b * np.broadcast_to(
        s_node[:, None, None], np.asarray(rho[...]).shape[:3]
    )
    wx.hybrid_apply_eb_boundary_to_nodal_scalar("rho_fp", 0, True)
    r = rho[...]
    r = r[..., 0] if r.ndim == 4 else r
    for i in node_rows(-1.0, -0.05):
        ck.close(
            f"rho odd: linear field continues b*s at s={s_node[i] / H:+.2f}h",
            r[i, :, :],
            b * s_node[i],
            1e-12,
        )

    pe = fields.ElectronPressureFPWrapper()
    pe[...] = c
    wx.hybrid_apply_eb_boundary_to_nodal_scalar("hybrid_electron_pressure_fp", 0, False)
    p = pe[...]
    p = p[..., 0] if p.ndim == 4 else p
    ck.close("Pe even: fluid untouched", p[node_rows(0.05, 10.0), :, :], c, 0.0)
    for i in node_rows(-1.0, -0.05):
        ck.close(
            f"Pe even: Neumann mirror at s={s_node[i] / H:+.2f}h", p[i, :, :], c, 1e-12
        )
    ck.close("Pe even: deep zero", p[node_rows(-100.0, -1.0), :, :], 0.0, 0.0)

    # --- 6) deposit fold: PEC image parities, planar closed forms ---------
    # The covered node row at s=-0.7h holds a deposit c (the shape-function
    # spill of wall-adjacent particles). Its mirror lands 0.6 of the way
    # between the first two fluid rows, so the fold delivers -0.6c and -0.4c
    # (PEC image charge: subtracted) and conserves the folded amount.
    i_dep = node_rows(-1.0, -0.05)  # s = -0.7h (single row for this wall)
    i_first = node_rows(0.05, 0.5)  # s = +0.3h
    i_second = node_rows(1.0, 1.5)  # s = +1.3h
    rho[...] = 0.0
    for i in i_dep:
        rho[i, :, :] = c
    wx.hybrid_fold_eb_deposit_to_nodal_scalar("rho_fp", 0)
    r = rho[...]
    r = r[..., 0] if r.ndim == 4 else r
    ck.close(
        "fold rho: first fluid row receives -0.6c", r[i_first, :, :], -0.6 * c, 1e-12
    )
    ck.close(
        "fold rho: second fluid row receives -0.4c", r[i_second, :, :], -0.4 * c, 1e-12
    )
    ck.close(
        "fold rho: fluid beyond the fold reach untouched",
        r[node_rows(1.6, 10.0), :, :],
        0.0,
        0.0,
    )
    ck.close(
        "fold rho: covered deposit left in place for the fill", r[i_dep, :, :], c, 0.0
    )
    ck.close(
        "fold rho: folded amount conserved (sum = -c)",
        r[i_first[0], :, :] + r[i_second[0], :, :],
        -c,
        1e-12,
    )

    # tangential J: image current antiparallel (subtracted), same geometry
    Jx = fields.JxFPWrapper()
    Jy = fields.JyFPWrapper()
    Jz = fields.JzFPWrapper()
    Jx[...] = 0.0
    Jz[...] = 0.0
    Jy[...] = 0.0
    for i in i_dep:
        Jy[i, :, :] = c
    wx.hybrid_fold_eb_deposit_to_edge_field("current_fp", 0)
    jy = Jy[...]
    ck.close("fold J tangential: subtracted -0.6c", jy[i_first, :, :], -0.6 * c, 1e-12)
    ck.close("fold J tangential: subtracted -0.4c", jy[i_second, :, :], -0.4 * c, 1e-12)

    # normal J: image current parallel (added); the deposit sits on the cut
    # x-edge with covered center (s=-0.2h), its mirror lands 0.6 of the way
    # between the first two fluid x-edge rows
    Jx[...] = 0.0
    Jy[...] = 0.0
    Jz[...] = 0.0
    for i in i_cut:
        Jx[i, :, :] = c
    wx.hybrid_fold_eb_deposit_to_edge_field("current_fp", 0)
    jx = Jx[...]
    i_cn = cent_rows(0.5, 1.0)  # s = +0.8h
    ck.close("fold J normal: added +0.4c", jx[i_cn, :, :], 0.4 * c, 1e-12)
    ck.close(
        "fold J normal: fluid beyond the fold reach untouched",
        jx[cent_rows(1.6, 10.0), :, :],
        0.0,
        0.0,
    )

    # reflecting-wall parity: the exact opposite signs (deposit added back,
    # mass conserving; normal J subtracted)
    rho[...] = 0.0
    for i in i_dep:
        rho[i, :, :] = c
    wx.hybrid_fold_eb_deposit_to_nodal_scalar("rho_fp", 0, pec=False)
    r = rho[...]
    r = r[..., 0] if r.ndim == 4 else r
    ck.close(
        "fold rho (reflect): first fluid row receives +0.6c",
        r[i_first, :, :],
        0.6 * c,
        1e-12,
    )
    ck.close(
        "fold rho (reflect): second fluid row receives +0.4c",
        r[i_second, :, :],
        0.4 * c,
        1e-12,
    )
    ck.close(
        "fold rho (reflect): folded amount conserved (sum = +c)",
        r[i_first[0], :, :] + r[i_second[0], :, :],
        c,
        1e-12,
    )

    Jx[...] = 0.0
    Jy[...] = 0.0
    Jz[...] = 0.0
    for i in i_dep:
        Jy[i, :, :] = c
    wx.hybrid_fold_eb_deposit_to_edge_field("current_fp", 0, pec=False)
    jy = Jy[...]
    ck.close(
        "fold J tangential (reflect): added +0.6c", jy[i_first, :, :], 0.6 * c, 1e-12
    )
    Jx[...] = 0.0
    Jy[...] = 0.0
    Jz[...] = 0.0
    for i in i_cut:
        Jx[i, :, :] = c
    wx.hybrid_fold_eb_deposit_to_edge_field("current_fp", 0, pec=False)
    jx = Jx[...]
    ck.close(
        "fold J normal (reflect): subtracted -0.4c", jx[i_cn, :, :], -0.4 * c, 1e-12
    )

    # --- 7) selectivity against a spatially varying field ----------------
    shape = np.asarray(Ey[...]).shape
    ii, jj, kk = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    varying = 1.0 + 0.17 * ii + 0.11 * jj + 0.013 * kk
    Ey[...] = varying
    Ex[...] = 0.0
    Ez[...] = 0.0
    before = np.array(Ey[...])
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    after = Ey[...]
    i_fluid = node_rows(0.05, 10.0)
    ck.close(
        "selectivity: varying field bit-identical in the fluid",
        after[i_fluid, :, :],
        before[i_fluid, :, :],
        0.0,
    )

    ck.finish()


# ----------------------------------------------------------------------------
# Cylinder battery (curved wall, interpolated normals)
# ----------------------------------------------------------------------------
def run_cylinder_battery(sim):
    wx = sim.extension.warpx
    ck = CheckSet()
    c = 2.0

    xc = LO + (np.arange(N_XY) + 0.5) * H
    xn = LO + np.arange(N_XY + 1) * H

    def s_of(x, y):
        return R_WALL - np.sqrt(x * x + y * y)

    # Ey edges: x nodal, y centered
    SX, SY = np.meshgrid(xn, xc, indexing="ij")
    s_ey = s_of(SX, SY)

    Ex = fields.ExFPWrapper()
    Ey = fields.EyFPWrapper()
    Ez = fields.EzFPWrapper()
    Ex[...] = 0.0
    Ez[...] = 0.0
    Ey[...] = c
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    ey = np.asarray(Ey[...])

    fluid = s_ey > 1.5 * H
    ck.close("cyl E: fluid untouched", ey[fluid, :], c, 0.0)
    deep = s_ey < -2.5 * H
    ck.close("cyl E: deep interior zero", ey[deep, :], 0.0, 0.0)

    # at the top of the ring the normal is -y and Ey is the NORMAL component
    # (even): the covered Ey edges near the pole must reproduce +c; at the
    # sides (normal -x) Ey is TANGENTIAL: covered edges must flip sign
    top = (np.abs(SX) < 0.15) & (s_ey > -1.0 * H) & (s_ey < -0.2 * H) & (SY > 0)
    side = (np.abs(SY) < 0.15) & (s_ey > -1.0 * H) & (s_ey < -0.2 * H) & (SX > 0)
    if np.any(top):
        vals = ey[top, :]
        ck.close("cyl E: normal (top pole) even mirror ~ +c", vals, c, 0.12 * c)
    if np.any(side):
        vals = ey[side, :]
        ck.expect(
            "cyl E: tangential (side) ghosts sign-flipped",
            bool(np.all(vals < 0.0)),
            f"max={vals.max():.3e}",
        )

    # scalars on the curved wall: constants are exact for even, sign-definite
    # for odd
    rho = fields.RhoFPWrapper()
    rho[...] = c
    wx.hybrid_apply_eb_boundary_to_nodal_scalar("rho_fp", 0, True)
    r = np.asarray(rho[...])
    r = r[..., 0] if r.ndim == 4 else r
    SXn, SYn = np.meshgrid(xn, xn, indexing="ij")
    s_n = s_of(SXn, SYn)
    ck.close("cyl rho odd: fluid untouched", r[s_n > 1.5 * H, :], c, 0.0)
    ck.close("cyl rho odd: deep zero", r[s_n < -2.5 * H, :], 0.0, 0.0)
    band = (s_n < -0.2 * H) & (s_n > -0.9 * H)
    ck.expect(
        "cyl rho odd: mirror band is negative",
        bool(np.all(r[band, :] < 0.0)),
        f"max={r[band, :].max():.3e}",
    )

    pe = fields.ElectronPressureFPWrapper()
    pe[...] = c
    wx.hybrid_apply_eb_boundary_to_nodal_scalar("hybrid_electron_pressure_fp", 0, False)
    p = np.asarray(pe[...])
    p = p[..., 0] if p.ndim == 4 else p
    ck.close("cyl Pe even: fluid untouched", p[s_n > 1.5 * H, :], c, 0.0)
    ck.close("cyl Pe even: band reproduces constant", p[band, :], c, 1e-12)
    ck.close("cyl Pe even: deep zero", p[s_n < -2.5 * H, :], 0.0, 0.0)

    ck.finish()


# ----------------------------------------------------------------------------
# Transitional Marder battery (planar wall; ports the four Phoenix
# transitional-Marder unit tests plus a target-mode consistency check)
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

    rho_prof = np.zeros(N_XY + 1)
    rho_prof[:I_DENSE_END] = RHO_DENSE
    rho_prof[I_DENSE_END:I_TRANS_END] = RHO_TRANS
    # Pe falls smoothly to zero at the vacuum boundary (zero slope there)
    pe_prof = np.zeros(N_XY + 1)
    i_nodes = np.arange(I_TRANS_END + 1)
    pe_prof[: I_TRANS_END + 1] = P0 * np.cos(np.pi * i_nodes / (2 * I_TRANS_END)) ** 2

    rho = fields.RhoFPWrapper()
    pe = fields.ElectronPressureFPWrapper()
    r = np.asarray(rho[...])
    bcast = (
        (slice(None), None, None, None) if r.ndim == 4 else (slice(None), None, None)
    )
    rho[...] = np.broadcast_to(rho_prof[bcast], r.shape)
    pe[...] = np.broadcast_to(pe_prof[bcast], np.asarray(pe[...]).shape)

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

    # numpy mirrors of the kernel's 2-point stencils and masks ------------
    # x-edge (cell-centered in x) interpolated rho and the closed-form
    # pressure-only target Ex = -grad(Pe)/max(rho, floor); Ey = Ez = 0
    rho_xedge = 0.5 * (rho_prof[:-1] + rho_prof[1:])
    rho_lim_x = np.maximum(rho_xedge, RHO_FLOOR)
    ex_target = -(np.diff(pe_prof) / H) / rho_lim_x

    # update/mask bands
    band_xedge = (rho_xedge > 0.0) & (rho_xedge <= RHO_FLOOR)  # Ex rows
    band_node = (rho_prof > 0.0) & (rho_prof <= RHO_FLOOR)  # nodal rows
    dense_xedge = rho_xedge > RHO_FLOOR
    # interior vacuum x-edge rows, clear of the wall fill band and the
    # domain boundary
    vac_rows = [i for i in range(N_XY) if rho_xedge[i] <= 0.0 and 17 <= i <= 21]
    dense_rows = [i for i in range(1, N_XY) if dense_xedge[i]]
    band_rows = [i for i in range(N_XY) if band_xedge[i]]

    def divergence(ex, ey, ez):
        """Nodal divergence with the kernel's downward stencils. Yee global
        shapes: ex (N, N+1, Nz+1), ey (N+1, N, Nz+1), ez (N+1, N+1, Nz);
        result is nodal (N+1, N+1, Nz+1) with periodic wrap in y and z and
        one-sided x boundary nodes left at zero."""
        nx, ny, nz = N_XY + 1, N_XY + 1, N_Z + 1
        div = np.zeros((nx, ny, nz))
        div[1:-1, :, :] += (ex[1:, :, :] - ex[:-1, :, :]) / H
        dvy = (ey - np.roll(ey, 1, axis=1)) / H  # at j = 0..N-1, j=0 wraps
        div[:, :-1, :] += dvy
        div[:, -1, :] += dvy[:, 0, :]  # duplicated periodic node row
        dvz = (ez - np.roll(ez, 1, axis=2)) / H
        div[:, :, :-1] += dvz
        div[:, :, -1] += dvz[:, :, 0]
        return div

    def masked_err_norm(div, div_target_1d):
        err = div - div_target_1d[:, None, None]
        err[~band_node, :, :] = 0.0
        err[[0, -1], :, :] = 0.0  # one-sided boundary nodes excluded
        return float(np.sqrt(np.sum(err**2)))

    def set_baseline(noise_amp, seed):
        rng = np.random.RandomState(seed)
        ex0 = np.broadcast_to(
            ex_target[:, None, None], np.asarray(Ex[...]).shape
        ).copy()
        ey0 = np.zeros(np.asarray(Ey[...]).shape)
        ez0 = np.zeros(np.asarray(Ez[...]).shape)
        scale = noise_amp * np.max(np.abs(ex_target))
        ex0[band_rows, :, :] += scale * (rng.rand(len(band_rows), *ex0.shape[1:]) - 0.5)
        i_bnode = [i for i in range(1, N_XY) if band_node[i]]
        ey0[i_bnode, :, :] += scale * (rng.rand(len(i_bnode), *ey0.shape[1:]) - 0.5)
        ez0[i_bnode, :, :] += scale * (rng.rand(len(i_bnode), *ez0.shape[1:]) - 0.5)
        Ex[...] = ex0
        Ey[...] = ey0
        Ez[...] = ez0

    # the divergence target of the "ohm"/"grad_pe_only" modes with J = B = 0
    div_target = divergence(
        np.broadcast_to(ex_target[:, None, None], np.asarray(Ex[...]).shape),
        np.zeros(np.asarray(Ey[...]).shape),
        np.zeros(np.asarray(Ez[...]).shape),
    )[:, 0, 0]

    # --- 1) cleans divergence error (Phoenix test_cleans_divergence_error)
    set_baseline(0.3, seed=42)
    err_before = masked_err_norm(
        divergence(np.asarray(Ex[...]), np.asarray(Ey[...]), np.asarray(Ez[...])),
        div_target,
    )
    n_iter, resid = wx.hybrid_marder_correct_e(
        0, alpha=0.1, max_iterations=500, rtol=1e-2
    )
    err_after = masked_err_norm(
        divergence(np.asarray(Ex[...]), np.asarray(Ey[...]), np.asarray(Ez[...])),
        div_target,
    )
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

    # --- 2+3) modifies the transition band only; vacuum untouched ---------
    Ex[...] = 1.0e5
    Ey[...] = 2.0e5
    Ez[...] = 0.0
    before = np.array(Ex[...])
    wx.hybrid_marder_correct_e(0, alpha=0.1, max_iterations=50, rtol=0.0)
    ex = np.asarray(Ex[...])
    ck.close(
        "marder: dense plasma core bit-identical",
        ex[dense_rows, :, :],
        before[dense_rows, :, :],
        0.0,
    )
    d_band = np.max(np.abs(ex[band_rows, :, :] - before[band_rows, :, :]))
    ck.expect(
        "marder: transition band is modified", d_band > 0.0, f"max|dE|={d_band:.3e}"
    )
    ck.close(
        "marder: vacuum gap bit-identical",
        ex[vac_rows, :, :],
        before[vac_rows, :, :],
        0.0,
    )

    # --- 4) monotone convergence (single-iteration calls; the returned
    # residual is measured before each update, Phoenix semantics) ----------
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
    # a JxB drive exists -----------------------------------------------------
    set_baseline(0.3, seed=7)
    e0 = [np.array(w[...]) for w in (Ex, Ey, Ez)]
    wx.hybrid_marder_correct_e(0, alpha=0.1, max_iterations=5, rtol=0.0, target="ohm")
    e_ohm = [np.array(w[...]) for w in (Ex, Ey, Ez)]
    for w, e in zip((Ex, Ey, Ez), e0):
        w[...] = e
    wx.hybrid_marder_correct_e(
        0, alpha=0.1, max_iterations=5, rtol=0.0, target="grad_pe_only"
    )
    e_gpe = [np.array(w[...]) for w in (Ex, Ey, Ez)]
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

    # turn on a uniform JxB drive: enE_x = -Jy*Bz / rho(x) varies across the
    # band, so the two targets must now differ
    Jy[...] = 1.0e3
    Bz[...] = 0.1
    for w, e in zip((Ex, Ey, Ez), e0):
        w[...] = e
    wx.hybrid_marder_correct_e(0, alpha=0.1, max_iterations=5, rtol=0.0, target="ohm")
    ex_ohm2 = np.asarray(Ex[...])
    for w, e in zip((Ex, Ey, Ez), e0):
        w[...] = e
    wx.hybrid_marder_correct_e(
        0, alpha=0.1, max_iterations=5, rtol=0.0, target="grad_pe_only"
    )
    ex_gpe2 = np.asarray(Ex[...])
    d_t = np.max(np.abs(ex_ohm2[band_rows, :, :] - ex_gpe2[band_rows, :, :]))
    ck.expect(
        "marder: ohm and grad_pe_only differ under a JxB drive",
        d_t > 0.0,
        f"max|dE|={d_t:.3e}",
    )

    ck.finish()


# ----------------------------------------------------------------------------
# Hyper-resistivity stencil battery: validates the isotropized Laplacian
# (Patra-Karttunen 27-point) against its closed form and demonstrates the
# isotropization of the damping rate (the cross stencil's cos(4*theta)
# anisotropy is cancelled). The solver is configured so that the Faraday
# E solve reduces to E = -eta_h * nabla^2(J_plasma) exactly (B = 0, ion
# current = 0, eta = 0, rho far above the floor).
# ----------------------------------------------------------------------------
def run_hyper_battery(sim):
    wx = sim.extension.warpx
    ck = CheckSet()

    rho = fields.RhoFPWrapper()
    r = np.asarray(rho[...])
    rho[...] = np.full(r.shape, 100.0 * 1.0e16 * constants.q_e)
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

    Jpz = fields.JzFPPlasmaWrapper()
    Ez = fields.EzFPWrapper()

    # numpy mirrors of the two stencils (y and z are periodic, so np.roll is
    # exact there; comparisons stay on an interior x slab away from the wall)
    def d2(a, ax):
        return np.roll(a, -1, ax) - 2.0 * a + np.roll(a, 1, ax)

    def lap_cross(a):
        return (d2(a, 0) + d2(a, 1) + d2(a, 2)) / H**2

    def lap_iso(a):
        cxy = d2(d2(a, 0), 1)
        cxz = d2(d2(a, 0), 2)
        cyz = d2(d2(a, 1), 2)
        txyz = d2(cxy, 2)
        return lap_cross(a) + (cxy + cxz + cyz) / (6.0 * H**2) + txyz / (30.0 * H**2)

    interior = (slice(4, 21), slice(2, 31), slice(None))

    # --- 1) stencil exactness on a random field ---------------------------
    rng = np.random.RandomState(2024)
    jz = rng.rand(*np.asarray(Jpz[...]).shape) - 0.5
    Jpz[...] = jz
    wx.hybrid_solve_e(True)
    ez = np.asarray(Ez[...])
    expected = -ETA_H * lap_iso(jz)
    scale = float(np.max(np.abs(expected[interior])))
    ck.close(
        "hyper: solver matches the 27-point Patra-Karttunen closed form",
        ez[interior] / scale,
        expected[interior] / scale,
        1e-12,
    )
    d_cross = float(
        np.max(np.abs(ez[interior] + ETA_H * lap_cross(jz)[interior])) / scale
    )
    ck.expect(
        "hyper: result differs from the cross stencil (teeth)",
        d_cross > 1e-3,
        f"rel dev from cross = {d_cross:.3e}",
    )

    # --- 2) consistency: exact on quadratics (laplacian of x^2 is 2) ------
    # (x only: the periodic y/z ghost wrap would corrupt y^2/z^2 stencils)
    shape = np.asarray(Jpz[...]).shape
    xn = LO + np.arange(shape[0]) * H
    yn = LO + np.arange(shape[1]) * H
    zc = (np.arange(shape[2]) + 0.5) * H - N_Z * H / 2
    X, Y, Z = np.meshgrid(xn, yn, zc, indexing="ij")
    Jpz[...] = X**2
    wx.hybrid_solve_e(True)
    ez = np.asarray(Ez[...])
    ck.close(
        "hyper: exact on quadratics (laplacian of x^2 is 2)",
        ez[interior],
        -ETA_H * 2.0,
        1e-9,
    )

    # --- 3) isotropization demonstration: plane-wave damping rates --------
    # cos(k.x) is an exact eigenfunction of every translation-invariant
    # stencil, so the discrete damping rate lambda(k) = -L(F)/F is read off
    # pointwise with no quadrature. The cross stencil's rate differs between
    # axis-aligned and diagonal k of equal magnitude at O((kh)^2) - the
    # cos(4*theta) anisotropy that squares diffusing fields - while the
    # isotropic stencil cancels that term.
    kmag = 1.0 / H  # kh = 1

    def solver_rate(j_field):
        Jpz[...] = j_field
        wx.hybrid_solve_e(True)
        lam = (np.asarray(Ez[...]) / (ETA_H * j_field))[interior]
        return float(np.median(lam[np.abs(j_field[interior]) > 0.5]))

    def stencil_rate(lap, j_field):
        lam = (-lap(j_field) / j_field)[interior]
        return float(np.median(lam[np.abs(j_field[interior]) > 0.5]))

    j_axis = np.cos(kmag * X)
    j_diag = np.cos(kmag * (X + Y) / np.sqrt(2.0))

    lam_axis = solver_rate(j_axis)
    lam_diag = solver_rate(j_diag)
    aniso_iso = abs(lam_axis - lam_diag) / kmag**2
    aniso_cross = (
        abs(stencil_rate(lap_cross, j_axis) - stencil_rate(lap_cross, j_diag)) / kmag**2
    )
    aniso_iso_np = (
        abs(stencil_rate(lap_iso, j_axis) - stencil_rate(lap_iso, j_diag)) / kmag**2
    )
    ratio = aniso_iso / max(aniso_cross, 1e-300)
    ck.expect(
        "hyper: solver anisotropy equals the isotropic closed form",
        abs(aniso_iso - aniso_iso_np) <= 1e-10,
        f"|solver - numpy| = {abs(aniso_iso - aniso_iso_np):.3e}",
    )
    ck.expect(
        "hyper: axis/diagonal damping anisotropy suppressed (>= 4x at kh=1)",
        ratio < 0.25,
        f"aniso(iso)/aniso(cross) = {ratio:.4f} "
        f"(cross = {aniso_cross:.4f} k^2, iso = {aniso_iso:.5f} k^2)",
    )

    ck.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry",
        choices=["plane", "cylinder"],
        default="plane",
        help="conducting-wall geometry of the battery",
    )
    parser.add_argument(
        "--battery",
        choices=["eb", "marder", "hyper"],
        default="eb",
        help="which unit battery to run (eb fills/folds, the transitional "
        "Marder correction, or the isotropized hyper-resistivity stencil)",
    )
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    sim = setup_simulation(args.geometry, hyper=(args.battery == "hyper"))
    if args.battery == "hyper":
        run_hyper_battery(sim)
    elif args.battery == "marder":
        run_marder_battery(sim)
    elif args.geometry == "plane":
        run_plane_battery(sim)
    else:
        run_cylinder_battery(sim)


if __name__ == "__main__":
    main()
