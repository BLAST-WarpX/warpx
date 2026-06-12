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

# mirror geometry of the fill operators (EBJBoundary.cpp)
D_IMG_MIN = 0.5 * H


def vector_ratio(s):
    """Odd-parity scaling s/d_im of the staggered vector fill."""
    offset = max(max(abs(s), D_IMG_MIN) - s, H)
    return s / (s + offset)


def scalar_ratio(s):
    """Odd-parity scaling s/d_im of the nodal scalar fill (exact mirror)."""
    return s / max(abs(s), D_IMG_MIN)


def setup_simulation(geometry):
    grid = picmi.Cartesian3DGrid(
        number_of_cells=[N_XY, N_XY, N_Z],
        lower_bound=[LO, LO, -N_Z * H / 2],
        upper_bound=[HI, HI, N_Z * H / 2],
        lower_boundary_conditions=["dirichlet", "periodic", "periodic"],
        upper_boundary_conditions=["dirichlet", "periodic", "periodic"],
        lower_boundary_conditions_particles=["absorbing", "periodic", "periodic"],
        upper_boundary_conditions_particles=["absorbing", "periodic", "periodic"],
        warpx_max_grid_size=2048,
        warpx_max_grid_size_x=N_XY // 2,
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
        plasma_resistivity=1.0e-6,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry",
        choices=["plane", "cylinder"],
        default="plane",
        help="conducting-wall geometry of the battery",
    )
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    sim = setup_simulation(args.geometry)
    if args.geometry == "plane":
        run_plane_battery(sim)
    else:
        run_cylinder_battery(sim)


if __name__ == "__main__":
    main()
