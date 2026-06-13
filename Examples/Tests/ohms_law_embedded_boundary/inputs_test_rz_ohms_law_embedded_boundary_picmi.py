#!/usr/bin/env python3
#
# --- RZ unit tests of the hybrid solver's embedded-boundary value
# --- application (the RZ counterpart of
# --- inputs_test_3d_ohms_law_embedded_boundary_picmi.py): a small (r, z)
# --- grid with a deterministic conducting wall is initialized, synthetic
# --- fields with known closed-form behavior are loaded through the field
# --- wrappers, the boundary operators are applied directly through their
# --- Python bindings, and the values are asserted point by point. The fill
# --- operates on the (r, z) level set with Euclidean mirror geometry in the
# --- r-z plane, so the mirror formulas are identical to the 3D battery.
# ---
# --- Planar-in-RZ wall (conductor z > Z_WALL, fluid normal -z_hat): Er and
# --- Etheta are TANGENTIAL (odd mirrors), Ez is NORMAL (even); for B the
# --- parity is swapped (Bz odd, Br/Btheta even). All normals are axis
# --- aligned, so the closed forms hold to round-off.
# ---
# --- One structural difference from the 3D battery: RZ has no conformal
# --- (ECT) update, so the eb_update masks come from the STAIRCASE marking,
# --- which also masks the staggered points one row into the fluid (their
# --- edges/faces touch the cut cell row). Those rows (s = +0.3h here) are
# --- fill targets too, and the covered row at s = -0.7h then has an
# --- ill-posed image stencil resolved by the deterministic cascade. The
# --- resulting closed forms (RATIO_FILL_FIRST/RATIO_FILL_BAND below) are
# --- still exact to round-off.
# ---
# --- Cone wall (conductor r > r0 + slope*z, cut cells with mixed r-z
# --- normals): sign/containment assertions only, like the 3D cylinder
# --- battery.

import argparse
import sys

import numpy as np

from pywarpx import fields, picmi

constants = picmi.constants

N_R = 16
N_Z = 32
R_MAX = 0.5
Z_LO = -0.5
Z_HI = 0.5
H = (Z_HI - Z_LO) / N_Z  # square cells: dr = dz = H

# planar wall: conductor z > Z_WALL, fluid-pointing normal -z_hat; the 0.3 h
# offset from a node puts a cut z-edge with a covered center next to the wall
Z_WALL_OFFSET = 0.3
Z_WALL_NODE = 24
Z_WALL = Z_LO + (Z_WALL_NODE + Z_WALL_OFFSET) * H

# cone variant: conductor r > CONE_R0 + CONE_SLOPE*z (sloped wall in r-z)
CONE_R0 = 0.35
CONE_SLOPE = 0.2

# mirror geometry of the fill operators (EBJBoundary.cpp)
D_IMG_MIN = 0.5 * H


def vector_ratio(s):
    """Odd-parity scaling s/d_im of the staggered vector fill."""
    offset = max(max(abs(s), D_IMG_MIN) - s, H)
    return s / (s + offset)


def scalar_ratio(s):
    """Odd-parity scaling s/d_im of the nodal scalar fill (exact mirror)."""
    return s / max(abs(s), D_IMG_MIN)


# Staircase-mask closed forms of the odd vector fill on this wall (see the
# header): the first masked row sits at s = +0.3h IN the fluid with image at
# s = +1.3h (a solution row), so it fills to (0.3/1.3)*c; the covered row at
# s = -0.7h is cascade-resolved, its image at s = +0.7h interpolating 0.6 of
# the first-row value and 0.4 of the solution value, scaled by s/d_im = -1:
#   v(-0.7h) = -(0.6*(0.3/1.3) + 0.4)*c = -(7/13)*c
RATIO_FILL_FIRST = vector_ratio(0.3 * H)  # = +3/13
RATIO_FILL_BAND = -(0.6 * RATIO_FILL_FIRST + 0.4)  # = -7/13


def setup_simulation(geometry, hyper=False):
    grid = picmi.CylindricalGrid(
        number_of_cells=[N_R, N_Z],
        lower_bound=[0.0, Z_LO],
        upper_bound=[R_MAX, Z_HI],
        n_azimuthal_modes=1,
        lower_boundary_conditions=["none", "dirichlet"],
        upper_boundary_conditions=["dirichlet", "dirichlet"],
        lower_boundary_conditions_particles=["none", "absorbing"],
        upper_boundary_conditions_particles=["absorbing", "absorbing"],
        warpx_max_grid_size=2048,
        # Split along z (the wall-normal direction) with the box seam at
        # j=24, i.e. right at the wall (j=24.3) and inside the fold reach:
        # this is a regression test for the seam-safe fold gathers (the
        # deposit-fold mirror gathers reach up to 2*fold_band = 3 cells past
        # a fold target and now read ghost-extended scratch copies, so the
        # closed forms below hold regardless of the box layout; with the
        # clamped in-place gathers of the original implementation this seam
        # delivered -c instead of -0.4c for the rho fold and nothing for the
        # J folds).
        # (the hyper battery needs a single box: the species-free deck
        # allocates the plasma current with zero ghost cells, so the
        # hyper-resistivity stencil must not straddle a box seam)
        warpx_max_grid_size_y=2048 if hyper else 24,
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

    # NOTE: use_conformal_eb is 3D-only (aborts in RZ) and is not passed
    sim.solver = picmi.HybridPICSolver(
        grid=grid,
        gamma=5.0 / 3.0,
        Te=0.0,
        n0=1.0e18,
        n_floor=1.0e16,
        # the hyper battery isolates the hyper-resistive term:
        # Ez = -eta_h * [ (1/r) d/dr(r dJz/dr) + d2Jz/dz2 ]
        plasma_resistivity=0.0 if hyper else 1.0e-6,
        plasma_hyper_resistivity=1.0 if hyper else None,
        substeps=4,
    )

    # warpx.eb_implicit_function is evaluated in RZ as f(x=r, y=0, z=z)
    # (WarpXInitEB.cpp ParserIF), so "z" is axial and "sqrt(x*x+y*y)" is r
    if geometry == "plane":
        sim.embedded_boundary = picmi.EmbeddedBoundary(
            implicit_function="(z-zw)", zw=Z_WALL
        )
    else:
        sim.embedded_boundary = picmi.EmbeddedBoundary(
            implicit_function="(sqrt(x*x+y*y)-(r0+slope*z))",
            r0=CONE_R0,
            slope=CONE_SLOPE,
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
    """Valid-domain (r, z) view of a field wrapper or raw global array; with
    n_azimuthal_modes=1 the RZ MultiFabs hold a single (m=0) component,
    dropped here if exposed as a trailing axis."""
    a = np.asarray(field[...])
    return a[:, :, 0] if a.ndim == 3 else a


def set_zprofile(wrapper, prof_z):
    """Load a pure-z profile, uniform in r (and in the mode component)."""
    shape = np.asarray(wrapper[...]).shape
    prof = prof_z[None, :] if len(shape) == 2 else prof_z[None, :, None]
    wrapper[...] = np.broadcast_to(prof, shape).copy()


def set_zrows(wrapper, rows, value):
    """Overwrite the given global z rows of a wrapper with a value."""
    a = np.array(np.asarray(wrapper[...]))
    a[:, rows, ...] = value
    wrapper[...] = a


# ----------------------------------------------------------------------------
# Planar-wall battery (round-off accurate)
# ----------------------------------------------------------------------------
def run_plane_battery(sim):
    wx = sim.extension.warpx
    ck = CheckSet()
    c = 2.0  # constant field amplitude
    b = 3.0  # linear field slope (per unit s)

    # staggered z coordinates and signed distances s = Z_WALL - z
    z_node = Z_LO + np.arange(N_Z + 1) * H
    z_cent = Z_LO + (np.arange(N_Z) + 0.5) * H
    s_node = Z_WALL - z_node
    s_cent = Z_WALL - z_cent

    def node_rows(lo, hi):
        return [j for j in range(N_Z + 1) if lo < s_node[j] / H <= hi]

    def cent_rows(lo, hi):
        return [j for j in range(N_Z) if lo < s_cent[j] / H <= hi]

    Er = fields.ExFPWrapper()  # (cc r, nodal z): tangential to the z wall
    Et = fields.EyFPWrapper()  # (nodal r, nodal z): out-of-plane, tangential
    Ez = fields.EzFPWrapper()  # (nodal r, cc z): normal to the z wall
    Br = fields.BxFPWrapper()  # (nodal r, cc z): tangential -> even fill
    Bt = fields.ByFPWrapper()  # (cc r, cc z): tangential -> even fill
    Bz = fields.BzFPWrapper()  # (cc r, nodal z): normal -> odd fill
    rho = fields.RhoFPWrapper()  # nodal (r, z)
    pe = fields.ElectronPressureFPWrapper()  # nodal (r, z)

    # --- 0) RZ wrapper shapes encode the Yee staggering (WarpX.cpp RZ
    # --- nodal flags); everything downstream indexes (r, z) with z = axis 1
    expected = {
        "Er (cc r, nodal z)": (Er, (N_R, N_Z + 1)),
        "Et (nodal r, nodal z)": (Et, (N_R + 1, N_Z + 1)),
        "Ez (nodal r, cc z)": (Ez, (N_R + 1, N_Z)),
        "Br (nodal r, cc z)": (Br, (N_R + 1, N_Z)),
        "Bt (cc r, cc z)": (Bt, (N_R, N_Z)),
        "Bz (cc r, nodal z)": (Bz, (N_R, N_Z + 1)),
        "rho (nodal r, nodal z)": (rho, (N_R + 1, N_Z + 1)),
        "Pe (nodal r, nodal z)": (pe, (N_R + 1, N_Z + 1)),
    }
    for name, (w, shape) in expected.items():
        got = arr2d(w).shape
        full = np.asarray(w[...]).shape
        ck.expect(f"staggering: {name} -> {shape}", got == shape, f"shape={full}")

    # --- 1) tangential E (Er, Etheta): odd mirror -------------------------
    # staircase nodal-z row layout: j<=23 (s>=+1.3h) solution; j=24
    # (s=+0.3h) masked fill row; j=25 (s=-0.7h) cascade-resolved; j>=26 deep
    Er[...] = c
    Et[...] = c
    Ez[...] = 0.0
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    er = arr2d(Er)
    et = arr2d(Et)

    j_fluid = node_rows(1.05, 10.0)  # solver-owned (solution) rows
    j_first = node_rows(0.05, 0.5)  # masked first row, s = +0.3h
    j_band = node_rows(-1.0, -0.05)  # covered cascade row, s = -0.7h
    j_deep = node_rows(-100.0, -1.0)
    ck.close("E tangential: Er solution rows untouched", er[:, j_fluid], c, 0.0)
    ck.close("E tangential: Et solution rows untouched", et[:, j_fluid], c, 0.0)
    ck.close(
        "E tangential: Er masked first row odd fill (+3/13)c",
        er[:, j_first],
        RATIO_FILL_FIRST * c,
        1e-12,
    )
    ck.close(
        "E tangential: Et masked first row odd fill (+3/13)c",
        et[:, j_first],
        RATIO_FILL_FIRST * c,
        1e-12,
    )
    ck.close(
        "E tangential: Er covered row cascade fill (-7/13)c",
        er[:, j_band],
        RATIO_FILL_BAND * c,
        1e-12,
    )
    ck.close(
        "E tangential: Et covered row cascade fill (-7/13)c",
        et[:, j_band],
        RATIO_FILL_BAND * c,
        1e-12,
    )
    ck.close("E tangential: Er deep interior zero", er[:, j_deep], 0.0, 0.0)
    ck.close("E tangential: Et deep interior zero", et[:, j_deep], 0.0, 0.0)

    # --- 2) tangential E: linear continuation through the surface ---------
    # Et = b*s in the fluid must continue as b*s through the filled rows
    # (the bilinear gather is exact on linear fields, both in the direct
    # fill of the first row and through the cascade): Dirichlet 0 at the wall
    set_zprofile(Et, b * s_node)
    Er[...] = 0.0
    Ez[...] = 0.0
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    et = arr2d(Et)
    for j in j_first + j_band:
        ck.close(
            f"E tangential: linear field continues b*s at s={s_node[j] / H:+.2f}h",
            et[:, j],
            b * s_node[j],
            1e-12,
        )

    # --- 3) normal E (Ez): even mirror + covered-center cut edge ----------
    Ez[...] = c
    Er[...] = 0.0
    Et[...] = 0.0
    # poke junk into the cut z-edge row whose center is covered
    # (edge [z_24, z_25] contains the wall; center s = (0.3-0.5)h = -0.2h)
    j_cut = cent_rows(-0.5, 0.0)
    set_zrows(Ez, j_cut, 1.0e6)
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    ez = arr2d(Ez)
    ck.close("E normal: fluid rows untouched", ez[:, cent_rows(0.05, 10.0)], c, 0.0)
    for j in j_cut:
        ck.close(
            f"E normal: covered-center cut edge filled (junk replaced) at s={s_cent[j] / H:+.2f}h",
            ez[:, j],
            c,
            1e-12,
        )
    for j in cent_rows(-1.0, -0.5):
        ck.close(
            f"E normal: even mirror at s={s_cent[j] / H:+.2f}h", ez[:, j], c, 1e-12
        )
    ck.close("E normal: deep interior zero", ez[:, cent_rows(-100.0, -1.0)], 0.0, 0.0)

    # --- 4) B: swapped parity (normal Bz odd, tangential Br/Bt even) ------
    # Bz faces live on the same nodal-z rows as Er/Et and the staircase
    # marking masks the same rows, so the odd closed forms are identical
    Br[...] = c
    Bt[...] = c
    Bz[...] = c
    wx.hybrid_apply_eb_boundary_to_face_field("Bfield_fp", 0)
    br = arr2d(Br)
    bt = arr2d(Bt)
    bz = arr2d(Bz)
    ck.close("B normal: solution rows untouched", bz[:, j_fluid], c, 0.0)
    ck.close(
        "B normal: masked first row odd fill (+3/13)c",
        bz[:, j_first],
        RATIO_FILL_FIRST * c,
        1e-12,
    )
    ck.close(
        "B normal: covered row cascade fill (-7/13)c",
        bz[:, j_band],
        RATIO_FILL_BAND * c,
        1e-12,
    )
    ck.close("B normal: deep zero", bz[:, j_deep], 0.0, 0.0)
    # cut Br/Bt rows (centers covered): without the 3D conformal (ECT)
    # update the staircase mask decides whether they are solver-owned
    # (untouched) or even-parity filled -- both reproduce the constant
    j_fc = cent_rows(0.05, 10.0)
    ck.close("B tangential: Br fluid rows untouched", br[:, j_fc], c, 0.0)
    ck.close("B tangential: Bt fluid rows untouched", bt[:, j_fc], c, 0.0)
    for j in j_cut:
        ck.close(
            f"B tangential: Br even at cut row s={s_cent[j] / H:+.2f}h",
            br[:, j],
            c,
            1e-12,
        )
        ck.close(
            f"B tangential: Bt even at cut row s={s_cent[j] / H:+.2f}h",
            bt[:, j],
            c,
            1e-12,
        )
    j_dc = cent_rows(-100.0, -1.0)
    ck.close("B tangential: Br deep zero", br[:, j_dc], 0.0, 0.0)
    ck.close("B tangential: Bt deep zero", bt[:, j_dc], 0.0, 0.0)

    # --- 5) nodal scalars: rho odd (Dirichlet 0), Pe even (Neumann) -------
    rho[...] = c
    wx.hybrid_apply_eb_boundary_to_nodal_scalar("rho_fp", 0, True)
    r = arr2d(rho)
    ck.close("rho odd: fluid untouched", r[:, node_rows(0.05, 10.0)], c, 0.0)
    for j in node_rows(-1.0, -0.05):
        ck.close(
            f"rho odd: mirror at s={s_node[j] / H:+.2f}h",
            r[:, j],
            scalar_ratio(s_node[j]) * c,
            1e-12,
        )
    ck.close("rho odd: deep zero", r[:, node_rows(-100.0, -1.0)], 0.0, 0.0)

    # linear rho = b*s: ghosts continue b*s exactly (zero at the surface)
    set_zprofile(rho, b * s_node)
    wx.hybrid_apply_eb_boundary_to_nodal_scalar("rho_fp", 0, True)
    r = arr2d(rho)
    for j in node_rows(-1.0, -0.05):
        ck.close(
            f"rho odd: linear field continues b*s at s={s_node[j] / H:+.2f}h",
            r[:, j],
            b * s_node[j],
            1e-12,
        )

    pe[...] = c
    wx.hybrid_apply_eb_boundary_to_nodal_scalar("hybrid_electron_pressure_fp", 0, False)
    p = arr2d(pe)
    ck.close("Pe even: fluid untouched", p[:, node_rows(0.05, 10.0)], c, 0.0)
    for j in node_rows(-1.0, -0.05):
        ck.close(
            f"Pe even: Neumann mirror at s={s_node[j] / H:+.2f}h", p[:, j], c, 1e-12
        )
    ck.close("Pe even: deep zero", p[:, node_rows(-100.0, -1.0)], 0.0, 0.0)

    # --- 6) deposit fold: PEC image parities, planar closed forms ----------
    # The covered node row at s=-0.7h holds a deposit c (the shape-function
    # spill of wall-adjacent particles). Its mirror lands 0.6 of the way
    # between the first two fluid rows, so the fold delivers -0.6c and -0.4c
    # (PEC image charge: subtracted) and conserves the folded amount.
    j_dep = j_band  # s = -0.7h (single covered row for this wall)
    j_second = node_rows(1.0, 1.5)  # s = +1.3h
    rho[...] = 0.0
    set_zrows(rho, j_dep, c)
    wx.hybrid_fold_eb_deposit_to_nodal_scalar("rho_fp", 0)
    r = arr2d(rho)
    ck.close("fold rho: first fluid row receives -0.6c", r[:, j_first], -0.6 * c, 1e-12)
    ck.close(
        "fold rho: second fluid row receives -0.4c", r[:, j_second], -0.4 * c, 1e-12
    )
    ck.close(
        "fold rho: fluid beyond the fold reach untouched",
        r[:, node_rows(1.6, 10.0)],
        0.0,
        0.0,
    )
    ck.close(
        "fold rho: covered deposit left in place for the fill", r[:, j_dep], c, 0.0
    )
    ck.close(
        "fold rho: folded amount conserved (sum = -c)",
        r[:, j_first[0]] + r[:, j_second[0]],
        -c,
        1e-12,
    )

    # tangential J: image current antiparallel (subtracted), same geometry;
    # unlike rho (whose fold targets every fluid node), the staggered fold
    # only writes solver-owned (S_SOLUTION) points, so under the staircase
    # masks the s=+0.3h row is skipped (it is a fill target, overwritten by
    # the boundary fill afterwards) and only the s=+1.3h row receives its
    # mirror weight
    Jr = fields.JxFPWrapper()
    Jt = fields.JyFPWrapper()
    Jz = fields.JzFPWrapper()
    Jr[...] = 0.0
    Jz[...] = 0.0
    Jt[...] = 0.0
    set_zrows(Jt, j_dep, c)
    wx.hybrid_fold_eb_deposit_to_edge_field("current_fp", 0)
    jt = arr2d(Jt)
    ck.close("fold J tangential: masked first row skipped", jt[:, j_first], 0.0, 0.0)
    ck.close("fold J tangential: subtracted -0.4c", jt[:, j_second], -0.4 * c, 1e-12)

    # normal J: image current parallel (added); the deposit sits on the cut
    # z-edge with covered center (s=-0.2h), its mirror lands 0.6 of the way
    # between the first two fluid z-edge rows
    Jr[...] = 0.0
    Jt[...] = 0.0
    Jz[...] = 0.0
    set_zrows(Jz, j_cut, c)
    wx.hybrid_fold_eb_deposit_to_edge_field("current_fp", 0)
    jz = arr2d(Jz)
    j_cn = cent_rows(0.5, 1.0)  # s = +0.8h
    ck.close("fold J normal: added +0.4c", jz[:, j_cn], 0.4 * c, 1e-12)
    ck.close(
        "fold J normal: fluid beyond the fold reach untouched",
        jz[:, cent_rows(1.6, 10.0)],
        0.0,
        0.0,
    )

    # reflecting-wall parity: the exact opposite signs (deposit added back,
    # mass conserving; normal J subtracted)
    rho[...] = 0.0
    set_zrows(rho, j_dep, c)
    wx.hybrid_fold_eb_deposit_to_nodal_scalar("rho_fp", 0, pec=False)
    r = arr2d(rho)
    ck.close(
        "fold rho (reflect): first fluid row receives +0.6c",
        r[:, j_first],
        0.6 * c,
        1e-12,
    )
    ck.close(
        "fold rho (reflect): second fluid row receives +0.4c",
        r[:, j_second],
        0.4 * c,
        1e-12,
    )
    ck.close(
        "fold rho (reflect): folded amount conserved (sum = +c)",
        r[:, j_first[0]] + r[:, j_second[0]],
        c,
        1e-12,
    )

    Jr[...] = 0.0
    Jt[...] = 0.0
    Jz[...] = 0.0
    set_zrows(Jt, j_dep, c)
    wx.hybrid_fold_eb_deposit_to_edge_field("current_fp", 0, pec=False)
    jt = arr2d(Jt)
    ck.close(
        "fold J tangential (reflect): added +0.4c", jt[:, j_second], 0.4 * c, 1e-12
    )
    Jr[...] = 0.0
    Jt[...] = 0.0
    Jz[...] = 0.0
    set_zrows(Jz, j_cut, c)
    wx.hybrid_fold_eb_deposit_to_edge_field("current_fp", 0, pec=False)
    jz = arr2d(Jz)
    ck.close("fold J normal (reflect): subtracted -0.4c", jz[:, j_cn], -0.4 * c, 1e-12)

    # --- 7) selectivity against a spatially varying field -----------------
    shape = np.asarray(Et[...]).shape
    idx = np.indices(shape)
    varying = 1.0 + 0.17 * idx[0] + 0.013 * idx[1]
    Et[...] = varying
    Er[...] = 0.0
    Ez[...] = 0.0
    before = np.array(np.asarray(Et[...]))
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    after = np.asarray(Et[...])
    ck.close(
        "selectivity: varying field bit-identical in the fluid",
        after[:, j_fluid],
        before[:, j_fluid],
        0.0,
    )

    ck.finish()


# ----------------------------------------------------------------------------
# Cone battery (sloped wall in r-z, mixed normals; sign/containment checks)
# ----------------------------------------------------------------------------
def run_cone_battery(sim):
    wx = sim.extension.warpx
    ck = CheckSet()
    c = 2.0

    r_node = np.arange(N_R + 1) * H
    z_node = Z_LO + np.arange(N_Z + 1) * H

    def s_of(r, z):
        """Exact signed distance to the cone line r = r0 + slope*z."""
        return (CONE_R0 + CONE_SLOPE * z - r) / np.sqrt(1.0 + CONE_SLOPE**2)

    # interior-z mask: the cone surface meets the z domain boundaries, where
    # the interpolated normal can be polluted by the level-set ghost rows;
    # band (normal-dependent) checks stay 2 cells clear of the z ends
    def z_interior(zg):
        return (zg > Z_LO + 2 * H) & (zg < Z_HI - 2 * H)

    # --- C1) tangential Etheta: containment + odd band sign ---------------
    # Etheta edges: nodal in r and z (out-of-plane: tangential everywhere)
    RN, ZN = np.meshgrid(r_node, z_node, indexing="ij")
    s_n = s_of(RN, ZN)

    Er = fields.ExFPWrapper()
    Et = fields.EyFPWrapper()
    Ez = fields.EzFPWrapper()
    Er[...] = 0.0
    Ez[...] = 0.0
    Et[...] = c
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    et = arr2d(Et)

    fluid = s_n > 1.5 * H
    ck.close("cone Et: fluid untouched", et[fluid], c, 0.0)
    deep = s_n < -2.5 * H
    ck.close("cone Et: deep interior zero", et[deep], 0.0, 0.0)
    band = (s_n < -0.2 * H) & (s_n > -0.9 * H) & z_interior(ZN)
    ck.expect(
        "cone Et: tangential ghost band sign-flipped",
        bool(np.all(et[band] < 0.0)),
        f"n={int(np.sum(band))} max={et[band].max():.3e}",
    )

    # --- C2) selectivity: fluid bit-identical against a varying field -----
    shape = np.asarray(Et[...]).shape
    idx = np.indices(shape)
    Et[...] = 1.0 + 0.17 * idx[0] + 0.013 * idx[1]
    before = np.array(np.asarray(Et[...]))
    wx.hybrid_apply_eb_boundary_to_edge_field("Efield_fp", 0)
    after = np.asarray(Et[...])
    ck.close(
        "cone Et: varying field bit-identical in the fluid",
        arr2d(after)[fluid],
        arr2d(before)[fluid],
        0.0,
    )

    # --- C3) nodal scalars on the sloped wall: constants exact for even,
    # --- sign-definite for odd
    rho = fields.RhoFPWrapper()
    rho[...] = c
    wx.hybrid_apply_eb_boundary_to_nodal_scalar("rho_fp", 0, True)
    r = arr2d(rho)
    ck.close("cone rho odd: fluid untouched", r[fluid], c, 0.0)
    ck.close("cone rho odd: deep zero", r[deep], 0.0, 0.0)
    ck.expect(
        "cone rho odd: mirror band is negative",
        bool(np.all(r[band] < 0.0)),
        f"max={r[band].max():.3e}",
    )

    pe = fields.ElectronPressureFPWrapper()
    pe[...] = c
    wx.hybrid_apply_eb_boundary_to_nodal_scalar("hybrid_electron_pressure_fp", 0, False)
    p = arr2d(pe)
    ck.close("cone Pe even: fluid untouched", p[fluid], c, 0.0)
    ck.close("cone Pe even: band reproduces constant", p[band], c, 1e-12)
    ck.close("cone Pe even: deep zero", p[deep], 0.0, 0.0)

    ck.finish()


# ----------------------------------------------------------------------------
# Hyper-resistivity axis battery: the cylindrical radial operator applied to
# the parabola Jz = C0 - C2*r^2 is exactly -4*C2 at every radius, INCLUDING
# the axis, where (1/r) d/dr(r dJz/dr) -> 2 d2Jz/dr2 by L'Hopital (the
# geometric term doubles the second derivative rather than cancelling).
# The discrete operator is exact on the parabola, so the check is round-off
# tight; the historical axis branch (a bare, ghost-dependent d2/dr2) returned
# half the correct value there.
# ----------------------------------------------------------------------------
def run_hyper_battery(sim):
    wx = sim.extension.warpx
    ck = CheckSet()
    ETA_H = 1.0
    C0, C2 = 2.0, 3.0

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

    Jpz = fields.JzFPPlasmaWrapper()
    Ez = fields.EzFPWrapper()

    # Jz/Ez are nodal in r: r_i = i*H
    shape = np.asarray(Jpz[...]).shape
    r = (np.arange(shape[0]) * H)[:, None]
    Jpz[...] = np.broadcast_to(C0 - C2 * r**2, shape)
    wx.hybrid_solve_e(True)
    ez = np.asarray(Ez[...])

    # interior: away from the z wall (node 24.3), the z domain rows and the
    # outer-r domain row; the axis row i=0 is the point of the test
    interior = (slice(0, N_R - 1), slice(2, 20))
    ck.close(
        "hyper RZ: parabola gives -4*C2 at every radius (axis included)",
        ez[interior],
        ETA_H * 4.0 * C2,
        1e-9,
    )
    ck.close(
        "hyper RZ: axis row carries the L'Hopital factor of two",
        ez[0, 2:20],
        ETA_H * 4.0 * C2,
        1e-9,
    )

    ck.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry",
        choices=["plane", "cone"],
        default="plane",
        help="conducting-wall geometry of the battery",
    )
    parser.add_argument(
        "--battery",
        choices=["eb", "hyper"],
        default="eb",
        help="which unit battery to run (eb fills/folds or the cylindrical "
        "hyper-resistivity axis operator)",
    )
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    sim = setup_simulation(args.geometry, hyper=(args.battery == "hyper"))
    if args.battery == "hyper":
        run_hyper_battery(sim)
    elif args.geometry == "plane":
        run_plane_battery(sim)
    else:
        run_cone_battery(sim)


if __name__ == "__main__":
    main()
