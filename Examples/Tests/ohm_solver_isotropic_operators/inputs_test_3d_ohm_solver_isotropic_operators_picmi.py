#!/usr/bin/env python3
#
# --- Self-asserting unit batteries of the isotropized hybrid Ohm's-law
# --- operators (hybrid_pic_model.isotropic_hyper_resistivity /
# --- isotropic_resistivity / isotropic_gradient): a small species-free grid
# --- is initialized, synthetic fields with known closed-form behavior are
# --- loaded through the field wrappers, the Ohm's-law solve is applied
# --- directly through its Python binding, and the values are asserted point
# --- by point against numpy mirrors of the stencils:
# ---
# ---   * each isotropized operator matches its closed form to round-off and
# ---     visibly differs from the standard (cross / two-point) stencil,
# ---   * the operators are pure truncation-error cancellers: exact (or zero)
# ---     on low-degree polynomials,
# ---   * plane waves at equal |k| along a grid axis and along the diagonal
# ---     quantify the cos(4*theta) anisotropy: the standard stencils damp /
# ---     scale the two directions differently at O((kh)^2), the isotropized
# ---     stencils suppress that axis/diagonal split by >= 4x at kh = 1,
# ---   * the isotropized gradient is NOT discretely curl-free (unlike the
# ---     plain staggered gradient, whose Yee curl vanishes identically); the
# ---     battery measures the O(h^2) curl defect against its closed form and
# ---     asserts the mitigation: the solve_for_Faraday E is bitwise
# ---     independent of the electron pressure, so the defect can never
# ---     generate magnetic field,
# ---   * with an embedded boundary present, the operators fall back to the
# ---     standard stencils within a corner reach of the level set and stay
# ---     isotropized in the deep fluid (eb_fallback battery).

import argparse
import os
import sys

import numpy as np

from pywarpx import fields, picmi

constants = picmi.constants

N_XY = 32
N_Z = 8
LO = -1.0
HI = 1.0
H = (HI - LO) / N_XY  # cubic cells

# planar wall of the eb_fallback battery: conductor x > X_WALL; the 0.3 h
# offset from a node puts the cut in the interior of cell X_WALL_NODE
X_WALL_OFFSET = 0.3
X_WALL_NODE = 24
X_WALL = LO + (X_WALL_NODE + X_WALL_OFFSET) * H

ETA_H = 1.0  # constant hyper-resistivity of the hyper battery
ETA_R = 1.0  # constant resistivity of the resistive battery
RHO0 = 100.0 * 1.0e16 * constants.q_e  # uniform density, well above n_floor


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


def setup_simulation(battery, grid_type="staggered"):
    # Species-free deck: the plasma current is allocated with zero ghost
    # cells, so the isotropization stencils must not straddle a box seam --
    # force a single box.
    hyper = battery.startswith("hyper") or battery == "eb_fallback"
    resistive = battery.startswith("resistive")
    gradient = battery.startswith("gradient") or battery == "eb_fallback"

    grid = picmi.Cartesian3DGrid(
        number_of_cells=[N_XY, N_XY, N_Z],
        lower_bound=[LO, LO, -N_Z * H / 2],
        upper_bound=[HI, HI, N_Z * H / 2],
        lower_boundary_conditions=["dirichlet", "periodic", "periodic"],
        upper_boundary_conditions=["dirichlet", "periodic", "periodic"],
        lower_boundary_conditions_particles=["absorbing", "periodic", "periodic"],
        upper_boundary_conditions_particles=["absorbing", "periodic", "periodic"],
        warpx_max_grid_size=2048,
        warpx_blocking_factor=8,
    )

    sim = picmi.Simulation(
        time_step_size=1.0e-9,
        max_steps=1,
        particle_shape=1,
        verbose=0,
    )
    sim.current_deposition_algo = "direct"
    sim.grid_type = grid_type

    sim.solver = picmi.HybridPICSolver(
        grid=grid,
        gamma=5.0 / 3.0,
        Te=0.0,
        n0=1.0e18,
        n_floor=1.0e16,
        # each battery isolates its own term: the hyper battery solves
        # E = -eta_h*nabla^2 J, the resistive battery E = eta*J plus the
        # corner-curl correction, the gradient battery E = -grad(Pe)/rho
        plasma_resistivity=(ETA_R if resistive else 0.0),
        plasma_hyper_resistivity=(ETA_H if hyper else None),
        isotropic_hyper_resistivity=(True if hyper else None),
        isotropic_resistivity=(True if resistive else None),
        isotropic_gradient=(True if gradient else None),
        substeps=4,
    )

    if battery == "eb_fallback":
        sim.embedded_boundary = picmi.EmbeddedBoundary(
            implicit_function="(x-xw)", xw=X_WALL
        )

    sim.initialize_inputs()
    sim.initialize_warpx()
    return sim


def zero_all_inputs():
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
        fields.ExFPWrapper(),
        fields.EyFPWrapper(),
        fields.EzFPWrapper(),
        fields.ElectronPressureFPWrapper(),
    ):
        w[...] = 0.0
    rho = fields.RhoFPWrapper()
    rho[...] = np.full(np.asarray(rho[...]).shape, RHO0)


# numpy mirrors of the stencils (y and z are periodic; comparisons stay on
# interior slices so the nodal-axis wrap duplication never enters a read)
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


# ----------------------------------------------------------------------------
# Hyper-resistivity battery (staggered): E = -eta_h * lap(J) with the
# isotropic 27-point Patra-Karttunen stencil
# ----------------------------------------------------------------------------
def run_hyper_battery(sim):
    wx = sim.extension.warpx
    ck = CheckSet()
    zero_all_inputs()

    Jpz = fields.JzFPPlasmaWrapper()
    Ez = fields.EzFPWrapper()

    interior = (slice(4, 29), slice(2, 31), slice(None))

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
    X = np.broadcast_to(xn[:, None, None], shape)
    Jpz[...] = X**2
    wx.hybrid_solve_e(True)
    ez = np.asarray(Ez[...])
    ck.close(
        "hyper: exact on quadratics (laplacian of x^2 is 2)",
        ez[interior],
        -ETA_H * 2.0,
        1e-9,
    )

    # --- 3) isotropization: plane-wave damping rates, axis vs diagonal ----
    # cos(k.x) is an exact eigenfunction of every translation-invariant
    # stencil, so the discrete damping rate lambda(k) = -L(F)/F is read off
    # pointwise with no quadrature. The cross stencil's rate differs between
    # axis-aligned and diagonal k of equal magnitude at O((kh)^2) -- the
    # cos(4*theta) anisotropy that squares diffusing fields -- while the
    # isotropic stencil cancels that term.
    yn = LO + np.arange(shape[1]) * H
    zc = (np.arange(shape[2]) + 0.5) * H - N_Z * H / 2
    X, Y, Z = np.meshgrid(xn, yn, zc, indexing="ij")
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


# ----------------------------------------------------------------------------
# Resistive isotropization battery (staggered): the corner-curl correction
# added to the resistive E makes the standard Faraday curl emit the in-plane
# Mehrstellen Laplacian of the out-of-plane B (Bz), cancelling the
# cross-stencil cos(4*theta) anisotropy that drives the grid m=4. Validated
# by isolating the correction (ion current zero, so the base eta*J term
# vanishes): curl_up(E) must equal -(eta/mu0)*corner(Bz) to round-off, the
# emergent operator (cross + corner) must be isotropic, div(B) must stay
# zero, and the correction must vanish for fields with no mixed 4th
# difference.
# ----------------------------------------------------------------------------
def run_resistive_battery(sim):
    wx = sim.extension.warpx
    ck = CheckSet()
    mu0 = constants.mu0
    zero_all_inputs()

    Bz = fields.BzFPWrapper()
    shp = np.asarray(Bz[...]).shape  # Bz: cc x, cc y, node z
    # common crop box (Ex,Ey,Bz have different staggered shapes); asserts use
    # the deep interior only, so cropping + periodic roll is safe there
    NC = (N_XY, N_XY, N_Z + 1)

    def crop(a):
        return np.asarray(a)[: NC[0], : NC[1], : NC[2]]

    def set_bz(prof2d):
        Bz[...] = np.broadcast_to(prof2d[:, :, None], shp).copy()

    def solve_corr():
        # ion current is zero, so E holds only the corner-curl correction
        wx.hybrid_solve_e(True)
        return crop(fields.ExFPWrapper()[...]), crop(fields.EyFPWrapper()[...])

    h = H

    def upx(F):
        return (np.roll(F, -1, 0) - F) / h

    def upy(F):
        return (np.roll(F, -1, 1) - F) / h

    ii = (slice(4, 20), slice(4, 20), slice(2, 4))

    # --- 1) the correction's Faraday curl is the Mehrstellen corner ---------
    rng = np.random.RandomState(7)
    bz = rng.rand(shp[0], shp[1]) - 0.5
    set_bz(bz)
    Ex, Ey = solve_corr()
    bz3 = np.broadcast_to(bz[:, :, None], shp)[: NC[0], : NC[1], : NC[2]]
    corner = (1.0 / (6.0 * h * h)) * d2(
        d2(bz3, 0), 1
    )  # cubic in-plane Mehrstellen corner
    curlz = upx(Ey) - upy(Ex)  # curl_up(E)_z
    expected = -(ETA_R / mu0) * corner
    scale = float(np.max(np.abs(expected[ii])))
    ck.close(
        "resistive: Faraday curl of the correction is the in-plane corner",
        curlz[ii] / scale,
        expected[ii] / scale,
        1e-12,
    )

    # --- 2) emergent operator (cross + corner) is isotropic; teeth ----------
    # the total resistive diffusion is (eta/mu0)(lap_cross + corner) = lap_iso
    lap_cross2 = (d2(bz3, 0) + d2(bz3, 1)) / h**2
    lap_iso2 = lap_cross2 + (1.0 / (6.0 * h * h)) * d2(d2(bz3, 0), 1)
    emergent = lap_cross2 - (mu0 / ETA_R) * curlz  # = lap_cross + corner_measured
    ck.close(
        "resistive: emergent operator equals the isotropic Laplacian",
        emergent[ii],
        lap_iso2[ii],
        1e-9 * float(np.max(np.abs(lap_iso2[ii]))),
    )
    teeth = float(np.max(np.abs(emergent[ii] - lap_cross2[ii])))
    ck.expect(
        "resistive: differs from the cross stencil (teeth)",
        teeth > 1e-3 * float(np.max(np.abs(lap_cross2[ii]))),
        f"max|iso-cross|={teeth:.3e}",
    )

    # --- 3) div(B) preserved exactly (correction enters only through E) -----
    # the Faraday increment of B from this E only changes Bz (dBz =
    # -dt*curl_up(E)_z), and for a z-independent dBz the discrete divergence
    # reduces to dnz(dBz) = 0 -> check round-off
    ck.expect(
        "resistive: div(B) increment is round-off",
        float(np.max(np.abs(np.roll(curlz, -1, 2)[ii] - curlz[ii]))) < 1e-6 * scale * h,
        "z-uniform dBz keeps div(B)=0",
    )

    # --- 4) plane-wave anisotropy of the emergent operator suppressed -------
    xc = (np.arange(NC[0]) + 0.5) * h + LO
    yc = (np.arange(NC[1]) + 0.5) * h + LO
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    kmag = 1.0 / h

    def rate_axis_diag(lap_fn):
        ja = np.cos(kmag * X)
        jd = np.cos(kmag * (X + Y) / np.sqrt(2.0))
        ja3 = np.broadcast_to(ja[:, :, None], NC)
        jd3 = np.broadcast_to(jd[:, :, None], NC)
        la = (-lap_fn(ja3) / ja3)[ii]
        ld = (-lap_fn(jd3) / jd3)[ii]
        return abs(float(np.median(la)) - float(np.median(ld))) / kmag**2

    def lap_cross_fn(a):
        return (d2(a, 0) + d2(a, 1)) / h**2

    def lap_iso_fn(a):
        return lap_cross_fn(a) + (1.0 / (6.0 * h * h)) * d2(d2(a, 0), 1)

    a_cross = rate_axis_diag(lap_cross_fn)
    a_iso = rate_axis_diag(lap_iso_fn)
    ck.expect(
        "resistive: axis/diagonal damping anisotropy suppressed (>= 4x at kh=1)",
        a_iso < 0.25 * a_cross,
        f"aniso(iso)/aniso(cross)={a_iso / max(a_cross, 1e-300):.4f}",
    )

    # --- 5) pure truncation canceller: zero on quadratics -------------------
    set_bz(X**2)
    Ex, Ey = solve_corr()
    ck.close(
        "resistive: correction vanishes on quadratics",
        np.asarray(Ex)[ii],
        0.0,
        1e-9 * float(np.max(np.abs(np.asarray(Ex)))) + 1e-12,
    )

    ck.finish()


# ----------------------------------------------------------------------------
# Gradient battery (staggered): E = -grad(Pe)/rho with the transverse-
# smoothed staggered gradient. Checks closed form, exactness on quadratics,
# axis/diagonal symbol anisotropy, the O(h^2) curl defect against its closed
# form, and the mitigation (the Faraday-path E is bitwise independent of Pe).
# ----------------------------------------------------------------------------
def run_gradient_battery(sim):
    wx = sim.extension.warpx
    ck = CheckSet()
    zero_all_inputs()

    Pe = fields.ElectronPressureFPWrapper()
    pe_shape = np.asarray(Pe[...]).shape  # nodal (N+1, N+1, N_Z+1)

    A_T = 1.0 / 24.0  # transverse weight of the staggered isotropized gradient

    def smooth(a, axes):
        out = a.copy()
        for ax in axes:
            out = out + A_T * d2(a, ax)
        return out

    def grad_x(a):  # plain staggered d/dx: nodal -> x-face
        return (a[1:, :, :] - a[:-1, :, :]) / H

    def grad_y(a):
        return (a[:, 1:, :] - a[:, :-1, :]) / H

    def grad_x_iso(a):
        return grad_x(smooth(a, (1, 2)))

    def grad_y_iso(a):
        return grad_y(smooth(a, (0, 2)))

    # interiors on the E-component staggerings (avoid the x dirichlet edges
    # and the periodic nodal wrap rows of the numpy mirrors)
    iix = (slice(3, 29), slice(2, 31), slice(1, N_Z))  # Ex: (N, N+1, N_Z+1)
    iiy = (slice(3, 30), slice(2, 30), slice(1, N_Z))  # Ey: (N+1, N, N_Z+1)

    def solve_grad():
        # solve_for_Faraday=False includes grad(Pe); with J = B = 0 and
        # uniform rho the solver E is exactly -grad(Pe)/RHO0
        wx.hybrid_solve_e(False)
        gx = -RHO0 * np.asarray(fields.ExFPWrapper()[...])
        gy = -RHO0 * np.asarray(fields.EyFPWrapper()[...])
        return gx, gy

    # --- 1) closed form on a random field, teeth vs the plain stencil -----
    rng = np.random.RandomState(11)
    pe = rng.rand(*pe_shape) - 0.5
    Pe[...] = pe
    gx, gy = solve_grad()
    scale = float(np.max(np.abs(grad_x_iso(pe)[iix])))
    ck.close(
        "gradient: solver matches the transverse-smoothed closed form (x)",
        gx[iix] / scale,
        grad_x_iso(pe)[iix] / scale,
        1e-12,
    )
    ck.close(
        "gradient: solver matches the transverse-smoothed closed form (y)",
        gy[iiy] / scale,
        grad_y_iso(pe)[iiy] / scale,
        1e-12,
    )
    d_plain = float(np.max(np.abs(gx[iix] - grad_x(pe)[iix])) / scale)
    ck.expect(
        "gradient: result differs from the two-point stencil (teeth)",
        d_plain > 1e-3,
        f"rel dev from plain = {d_plain:.3e}",
    )

    # --- 2) consistency: exact on quadratics (d/dx of x^2) ----------------
    xn = LO + np.arange(pe_shape[0]) * H
    Xn = np.broadcast_to(xn[:, None, None], pe_shape)
    Pe[...] = Xn**2
    gx, _ = solve_grad()
    xf = LO + (np.arange(pe_shape[0] - 1) + 0.5) * H  # x-face centers
    ck.close(
        "gradient: exact on quadratics (d/dx of x^2 at the face)",
        gx[iix],
        (2.0 * np.broadcast_to(xf[:, None, None], gx.shape))[iix],
        1e-9,
    )

    # --- 3) isotropization: symbol error, axis vs diagonal ----------------
    # For Pe = sin(k.x) the discrete gradient is ghat * cos(k.x) evaluated at
    # the (staggered) face centers, so ghat is read off pointwise. The plain
    # stencil's relative symbol error ghat_x/k_x - 1 is -(k_x h)^2/24 + ...:
    # it differs between an axis-aligned and a diagonal k of equal |k| at
    # O((kh)^2) (the cos(4*theta) anisotropy of |grad Pe|), while the
    # isotropized stencil's error depends on |k| only.
    yn = LO + np.arange(pe_shape[1]) * H
    zn = np.arange(pe_shape[2]) * H  # only relative spacing matters
    Xg, Yg, Zg = np.meshgrid(xn, yn, zn, indexing="ij")
    Xf, Yf_at_xf, Zf_at_xf = np.meshgrid(xf, yn, zn, indexing="ij")
    kmag = 1.0 / H  # kh = 1

    def symbol_x(kx, ky):
        """Median of G_x / (kx * cos) at the x-face centers: the discrete
        symbol of the solver's d/dx, normalized by the analytic one."""
        Pe[...] = np.sin(kx * Xg + ky * Yg)
        gx, _ = solve_grad()
        cosf = np.cos(kx * Xf + ky * Yf_at_xf)
        ratio = (gx / (kx * cosf))[iix]
        return float(np.median(ratio[np.abs(cosf[iix]) > 0.5]))

    def symbol_x_np(grad_fn, kx, ky):
        g = grad_fn(np.sin(kx * Xg + ky * Yg))
        cosf = np.cos(kx * Xf + ky * Yf_at_xf)
        ratio = (g / (kx * cosf))[iix]
        return float(np.median(ratio[np.abs(cosf[iix]) > 0.5]))

    err_axis = symbol_x(kmag, 0.0) - 1.0
    err_diag = symbol_x(kmag / np.sqrt(2.0), kmag / np.sqrt(2.0)) - 1.0
    aniso_iso = abs(err_axis - err_diag)
    err_axis_p = symbol_x_np(grad_x, kmag, 0.0) - 1.0
    err_diag_p = symbol_x_np(grad_x, kmag / np.sqrt(2.0), kmag / np.sqrt(2.0)) - 1.0
    aniso_plain = abs(err_axis_p - err_diag_p)
    ratio = aniso_iso / max(aniso_plain, 1e-300)
    ck.expect(
        "gradient: axis error unchanged (isotropization equalizes, not improves)",
        abs(err_axis - err_axis_p) <= 1e-10,
        f"axis rel err iso = {err_axis:.5f}, plain = {err_axis_p:.5f}",
    )
    ck.expect(
        "gradient: axis/diagonal symbol anisotropy suppressed (>= 4x at kh=1)",
        ratio < 0.25,
        f"aniso(iso)/aniso(plain) = {ratio:.4f} "
        f"(plain = {aniso_plain:.5f}, iso = {aniso_iso:.6f})",
    )

    # --- 4) the curl defect: O(h^2)-small, matches its closed form --------
    # The plain staggered gradient is discretely curl-free (identically); the
    # per-component transverse smoothers break that identity by
    # D_x D_y (T_y - T_x) Pe. Use an asymmetric wave (the defect vanishes by
    # symmetry on the exact diagonal).
    kx, ky = kmag, 0.5 * kmag
    pe_wave = np.sin(kx * Xg + ky * Yg)
    Pe[...] = pe_wave
    gx, gy = solve_grad()
    NCk = (N_XY, N_XY, N_Z + 1)

    def crop(a):
        return np.asarray(a)[: NCk[0], : NCk[1], : NCk[2]]

    def upx(F):
        return (np.roll(F, -1, 0) - F) / H

    def upy(F):
        return (np.roll(F, -1, 1) - F) / H

    # pad the staggered mirrors back to a common crop for the curl
    gx_c = crop(np.concatenate([gx, gx[-1:, :, :]], axis=0))
    gy_c = crop(np.concatenate([gy, gy[:, -1:, :]], axis=1))
    curl_solver = upx(gy_c) - upy(gx_c)
    gx_np = crop(np.concatenate([grad_x_iso(pe_wave), gx[-1:, :, :]], axis=0))
    gy_np = crop(np.concatenate([grad_y_iso(pe_wave), gy[:, -1:, :]], axis=1))
    curl_np = upx(gy_np) - upy(gx_np)
    gx_p = crop(np.concatenate([grad_x(pe_wave), gx[-1:, :, :]], axis=0))
    gy_p = crop(np.concatenate([grad_y(pe_wave), gy[:, -1:, :]], axis=1))
    curl_plain = upx(gy_p) - upy(gx_p)
    iic = (slice(4, 20), slice(4, 20), slice(2, 4))
    gscale = kmag  # |grad| scale of the unit-amplitude wave
    ck.close(
        "gradient: curl defect matches its closed form",
        curl_solver[iic] / (kmag * gscale),
        curl_np[iic] / (kmag * gscale),
        1e-12,
    )
    ck.expect(
        "gradient: plain staggered gradient is discretely curl-free",
        float(np.max(np.abs(curl_plain[iic]))) < 1e-10 * kmag * gscale,
        f"max|curl(plain)| = {float(np.max(np.abs(curl_plain[iic]))):.3e}",
    )
    curl_rel = float(np.max(np.abs(curl_solver[iic]))) / (kmag * gscale)
    ck.expect(
        "gradient: curl defect is O(h^2)-small at kh=1",
        0.0 < curl_rel < 0.05,
        f"max|curl|/(k*|grad|) = {curl_rel:.4f}",
    )

    # --- 5) mitigation: the Faraday-path E is independent of Pe -----------
    # grad(Pe) is only evaluated when solve_for_Faraday=False, so the E that
    # feeds the B integration can never see the curl defect.
    Pe[...] = pe_wave
    wx.hybrid_solve_e(True)
    e_a = [
        np.asarray(w[...]).copy()
        for w in (fields.ExFPWrapper(), fields.EyFPWrapper(), fields.EzFPWrapper())
    ]
    Pe[...] = 5.0 + 3.0 * (rng.rand(*pe_shape) - 0.5)
    wx.hybrid_solve_e(True)
    e_b = [
        np.asarray(w[...]).copy()
        for w in (fields.ExFPWrapper(), fields.EyFPWrapper(), fields.EzFPWrapper())
    ]
    dmax = max(float(np.max(np.abs(a - b))) for a, b in zip(e_a, e_b))
    ck.expect(
        "gradient: solve_for_Faraday E is bitwise independent of Pe",
        dmax == 0.0,
        f"max|dE| = {dmax:.3e}",
    )

    ck.finish()


# ----------------------------------------------------------------------------
# Collocated (nodal) batteries: every component lives at the node, the
# Faraday/gradient differences are the wide centered ones, and the
# isotropization operators use their nodal variants.
# ----------------------------------------------------------------------------
def run_hyper_collocated_battery(sim):
    """Nodal iso-hyper: E == -eta_h*lap_iso(Jz) (27-pt Patra-Karttunen) to
    round-off, differs from the cross stencil, axis/diagonal anisotropy killed."""
    wx = sim.extension.warpx
    ck = CheckSet()
    zero_all_inputs()

    Jpz = fields.JzFPPlasmaWrapper()
    Ez = fields.EzFPWrapper()

    interior = (slice(4, 29), slice(2, 31), slice(2, N_Z - 1))
    shape = np.asarray(Jpz[...]).shape

    rng = np.random.RandomState(2024)
    jz = rng.rand(*shape) - 0.5
    Jpz[...] = jz
    wx.hybrid_solve_e(True)
    ez = np.asarray(Ez[...])
    expected = -ETA_H * lap_iso(jz)
    scale = float(np.max(np.abs(expected[interior])))
    ck.close(
        "nodal hyper: matches 27-pt Patra-Karttunen closed form",
        ez[interior] / scale,
        expected[interior] / scale,
        1e-12,
    )
    d_cross = float(
        np.max(np.abs(ez[interior] + ETA_H * lap_cross(jz)[interior])) / scale
    )
    ck.expect(
        "nodal hyper: differs from the cross stencil (teeth)",
        d_cross > 1e-3,
        f"rel dev = {d_cross:.3e}",
    )

    # plane-wave anisotropy suppression
    xn = LO + np.arange(shape[0]) * H
    yn = LO + np.arange(shape[1]) * H
    zn = np.arange(shape[2]) * H
    X, Y, Z = np.meshgrid(xn, yn, zn, indexing="ij")
    kmag = 1.0 / H

    def solver_rate(jf):
        Jpz[...] = jf
        wx.hybrid_solve_e(True)
        lam = (np.asarray(Ez[...]) / (ETA_H * jf))[interior]
        return float(np.median(lam[np.abs(jf[interior]) > 0.5]))

    def stencil_rate(lap, jf):
        lam = (-lap(jf) / jf)[interior]
        return float(np.median(lam[np.abs(jf[interior]) > 0.5]))

    ja = np.cos(kmag * X)
    jd = np.cos(kmag * (X + Y) / np.sqrt(2.0))
    aniso_iso = abs(solver_rate(ja) - solver_rate(jd)) / kmag**2
    aniso_cross = (
        abs(stencil_rate(lap_cross, ja) - stencil_rate(lap_cross, jd)) / kmag**2
    )
    ck.expect(
        "nodal hyper: axis/diagonal damping anisotropy suppressed (>= 4x at kh=1)",
        aniso_iso < 0.25 * aniso_cross,
        f"aniso(iso)/aniso(cross) = {aniso_iso / max(aniso_cross, 1e-300):.4f}",
    )

    ck.finish()


def run_resistive_collocated_battery(sim):
    """Nodal iso-resistivity corner-curl: with all currents zero the solver E is
    only the corner correction; it matches the derived nodal stencil to round-off,
    the emergent operator is isotropic, and it vanishes on quadratics."""
    wx = sim.extension.warpx
    ck = CheckSet()
    mu0 = constants.mu0
    zero_all_inputs()

    Bz = fields.BzFPWrapper()
    shp = np.asarray(Bz[...]).shape

    def set_bz(prof2d):
        Bz[...] = np.broadcast_to(prof2d[:, :, None], shp).copy()

    def Dn(F, ax):
        return (np.roll(F, -1, ax) - np.roll(F, 1, ax)) / (2.0 * H)

    ii = (slice(5, 27), slice(5, 28), slice(2, N_Z - 1))

    rng = np.random.RandomState(7)
    bz2 = rng.rand(shp[0], shp[1]) - 0.5
    set_bz(bz2)
    wx.hybrid_solve_e(True)
    Ex = np.asarray(fields.ExFPWrapper()[...])
    Ey = np.asarray(fields.EyFPWrapper()[...])
    bz3 = np.broadcast_to(bz2[:, :, None], shp)
    Ex_expect = ETA_R * (1.0 / mu0) * (1.0 / 3.0) * Dn(d2(bz3, 0), 1)
    Ey_expect = -ETA_R * (1.0 / mu0) * (1.0 / 3.0) * Dn(d2(bz3, 1), 0)
    sx = float(np.max(np.abs(Ex_expect[ii]))) + 1e-300
    sy = float(np.max(np.abs(Ey_expect[ii]))) + 1e-300
    ck.close(
        "nodal resistive: Ex matches the derived corner stencil",
        Ex[ii] / sx,
        Ex_expect[ii] / sx,
        1e-12,
    )
    ck.close(
        "nodal resistive: Ey matches the derived corner stencil",
        Ey[ii] / sy,
        Ey_expect[ii] / sy,
        1e-12,
    )

    # emergent operator (wide cross + corner) isotropic vs the uncorrected cross
    xs = np.arange(shp[0]) * H
    ys = np.arange(shp[1]) * H
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    kmag = 1.0 / H

    def wide_cross(F):
        return Dn(Dn(F, 0), 0) + Dn(Dn(F, 1), 1)

    def emergent_rate(prof2d):
        set_bz(prof2d)
        wx.hybrid_solve_e(True)
        ex = np.asarray(fields.ExFPWrapper()[...])
        ey = np.asarray(fields.EyFPWrapper()[...])
        b3 = np.broadcast_to(prof2d[:, :, None], shp)
        op = wide_cross(b3) + (mu0 / ETA_R) * (Dn(ex, 1) - Dn(ey, 0))
        lam = (op / b3)[ii]
        return float(np.median(lam[np.abs(b3[ii]) > 0.5]))

    def wide_rate(prof2d):
        b3 = np.broadcast_to(prof2d[:, :, None], shp)
        lam = (wide_cross(b3) / b3)[ii]
        return float(np.median(lam[np.abs(b3[ii]) > 0.5]))

    ja = np.cos(kmag * X)
    jd = np.cos(kmag * (X + Y) / np.sqrt(2.0))
    aniso_iso = abs(emergent_rate(ja) - emergent_rate(jd)) / kmag**2
    aniso_cross = abs(wide_rate(ja) - wide_rate(jd)) / kmag**2
    ck.expect(
        "nodal resistive: axis/diagonal anisotropy suppressed (>= 4x at kh=1)",
        aniso_iso < 0.25 * aniso_cross,
        f"aniso(iso)/aniso(cross) = {aniso_iso / max(aniso_cross, 1e-300):.4f}",
    )

    set_bz(X**2)
    wx.hybrid_solve_e(True)
    ex = np.asarray(fields.ExFPWrapper()[...])
    ck.close(
        "nodal resistive: correction vanishes on quadratics",
        ex[ii],
        0.0,
        1e-9 * float(np.max(np.abs(ex))) + 1e-12,
    )

    ck.finish()


def run_gradient_collocated_battery(sim):
    """Nodal iso-gradient: wide centered difference with transverse weight
    1/6; closed form to round-off and axis/diagonal anisotropy suppressed."""
    wx = sim.extension.warpx
    ck = CheckSet()
    zero_all_inputs()

    Pe = fields.ElectronPressureFPWrapper()
    pe_shape = np.asarray(Pe[...]).shape  # fully nodal

    A_T = 1.0 / 6.0  # transverse weight of the nodal isotropized gradient

    def smooth(a, axes):
        out = a.copy()
        for ax in axes:
            out = out + A_T * d2(a, ax)
        return out

    def grad_x_n(a):  # wide centered difference, node -> node
        return (np.roll(a, -1, 0) - np.roll(a, 1, 0)) / (2.0 * H)

    def grad_x_iso_n(a):
        return grad_x_n(smooth(a, (1, 2)))

    ii = (slice(4, 29), slice(2, 31), slice(2, N_Z - 1))

    def solve_grad_x():
        wx.hybrid_solve_e(False)
        return -RHO0 * np.asarray(fields.ExFPWrapper()[...])

    rng = np.random.RandomState(11)
    pe = rng.rand(*pe_shape) - 0.5
    Pe[...] = pe
    gx = solve_grad_x()
    scale = float(np.max(np.abs(grad_x_iso_n(pe)[ii])))
    ck.close(
        "nodal gradient: solver matches the transverse-smoothed closed form",
        gx[ii] / scale,
        grad_x_iso_n(pe)[ii] / scale,
        1e-12,
    )
    d_plain = float(np.max(np.abs(gx[ii] - grad_x_n(pe)[ii])) / scale)
    ck.expect(
        "nodal gradient: differs from the wide centered stencil (teeth)",
        d_plain > 1e-3,
        f"rel dev = {d_plain:.3e}",
    )

    # axis/diagonal symbol anisotropy (nodal symbols are read off at nodes)
    xn = LO + np.arange(pe_shape[0]) * H
    yn = LO + np.arange(pe_shape[1]) * H
    zn = np.arange(pe_shape[2]) * H
    Xg, Yg, Zg = np.meshgrid(xn, yn, zn, indexing="ij")
    kmag = 1.0 / H

    def symbol_x(kx, ky, grad_fn=None):
        pe_w = np.sin(kx * Xg + ky * Yg)
        if grad_fn is None:
            Pe[...] = pe_w
            g = solve_grad_x()
        else:
            g = grad_fn(pe_w)
        cosn = np.cos(kx * Xg + ky * Yg)
        ratio = (g / (kx * cosn))[ii]
        return float(np.median(ratio[np.abs(cosn[ii]) > 0.5]))

    err_axis = symbol_x(kmag, 0.0) - 1.0
    err_diag = symbol_x(kmag / np.sqrt(2.0), kmag / np.sqrt(2.0)) - 1.0
    err_axis_p = symbol_x(kmag, 0.0, grad_x_n) - 1.0
    err_diag_p = symbol_x(kmag / np.sqrt(2.0), kmag / np.sqrt(2.0), grad_x_n) - 1.0
    aniso_iso = abs(err_axis - err_diag)
    aniso_plain = abs(err_axis_p - err_diag_p)
    ck.expect(
        "nodal gradient: axis/diagonal symbol anisotropy suppressed (>= 4x at kh=1)",
        aniso_iso < 0.25 * aniso_plain,
        f"aniso(iso)/aniso(plain) = {aniso_iso / max(aniso_plain, 1e-300):.4f} "
        f"(plain = {aniso_plain:.5f}, iso = {aniso_iso:.6f})",
    )

    ck.finish()


# ----------------------------------------------------------------------------
# EB fallback battery (staggered, planar wall): within a corner reach of the
# level set the isotropic operators fall back to the standard stencils; in
# the deep fluid they stay isotropized.
# ----------------------------------------------------------------------------
def run_eb_fallback_battery(sim):
    wx = sim.extension.warpx
    ck = CheckSet()
    zero_all_inputs()

    d_iso_compact = (np.sqrt(3.0) + 0.5) * H  # must match HybridPICSolveE.cpp

    # --- hyper-resistivity fallback on Ez (nodal x: phi sampled at nodes) --
    Jpz = fields.JzFPPlasmaWrapper()
    Ez = fields.EzFPWrapper()
    shape = np.asarray(Jpz[...]).shape

    rng = np.random.RandomState(2024)
    jz = rng.rand(*shape) - 0.5
    Jpz[...] = jz
    wx.hybrid_solve_e(True)
    ez = np.asarray(Ez[...])

    exp_iso = -ETA_H * lap_iso(jz)
    exp_cross = -ETA_H * lap_cross(jz)
    xn = LO + np.arange(shape[0]) * H
    phi = X_WALL - xn  # signed distance of the planar wall at the x-nodes
    # staircase masks own everything at least two cells from the cut cell;
    # restrict to rows that are BOTH solver-owned and mirror-valid
    i_iso = [i for i in range(4, shape[0]) if phi[i] >= d_iso_compact]
    i_compact = [i for i in range(4, shape[0]) if H <= phi[i] < d_iso_compact]
    yy = slice(2, 31)
    scale = float(np.max(np.abs(exp_iso[4:29, yy, :])))
    ck.expect(
        "eb_fallback: geometry provides both deep-fluid and near-wall rows",
        len(i_iso) >= 10 and len(i_compact) >= 1,
        f"iso rows = {len(i_iso)}, compact rows = {i_compact}",
    )
    ck.close(
        "eb_fallback: deep fluid uses the isotropic Laplacian",
        ez[i_iso, yy, :] / scale,
        exp_iso[i_iso, yy, :] / scale,
        1e-12,
    )
    ck.close(
        "eb_fallback: near-wall band falls back to the cross stencil",
        ez[i_compact, yy, :] / scale,
        exp_cross[i_compact, yy, :] / scale,
        1e-12,
    )

    # --- gradient fallback on Ex (cc x: phi sampled at the low node) ------
    Pe = fields.ElectronPressureFPWrapper()
    pe_shape = np.asarray(Pe[...]).shape
    pe = rng.rand(*pe_shape) - 0.5
    Pe[...] = pe
    wx.hybrid_solve_e(False)
    gx = -RHO0 * np.asarray(fields.ExFPWrapper()[...])

    A_T = 1.0 / 24.0

    def smooth(a, axes):
        out = a.copy()
        for ax in axes:
            out = out + A_T * d2(a, ax)
        return out

    def grad_x(a):
        return (a[1:, :, :] - a[:-1, :, :]) / H

    gx_iso = grad_x(smooth(pe, (1, 2)))
    gx_plain = grad_x(pe)
    # Ex face i sits between nodes i and i+1; the solver samples phi at the
    # kernel index (node i)
    i_iso_f = [i for i in range(4, gx.shape[0]) if phi[i] >= d_iso_compact]
    i_compact_f = [
        i for i in range(4, gx.shape[0]) if H <= phi[i + 1] and phi[i] < d_iso_compact
    ]
    gscale = float(np.max(np.abs(gx_iso[4:28, yy, 1:N_Z])))
    ck.close(
        "eb_fallback: deep-fluid gradient is isotropized",
        gx[i_iso_f, yy, 1:N_Z] / gscale,
        gx_iso[i_iso_f, yy, 1:N_Z] / gscale,
        1e-12,
    )
    if len(i_compact_f) > 0:
        ck.close(
            "eb_fallback: near-wall gradient falls back to the two-point stencil",
            gx[i_compact_f, yy, 1:N_Z] / gscale,
            gx_plain[i_compact_f, yy, 1:N_Z] / gscale,
            1e-12,
        )

    ck.finish()


BATTERIES = {
    "hyper": ("staggered", run_hyper_battery),
    "resistive": ("staggered", run_resistive_battery),
    "gradient": ("staggered", run_gradient_battery),
    "eb_fallback": ("staggered", run_eb_fallback_battery),
    "hyper_nodal": ("collocated", run_hyper_collocated_battery),
    "resistive_nodal": ("collocated", run_resistive_collocated_battery),
    "gradient_nodal": ("collocated", run_gradient_collocated_battery),
}

SUITES = {
    # eb_fallback is registered as its own CTest entry (it needs WarpX_EB)
    "staggered": ("hyper", "resistive", "gradient"),
    "collocated": ("hyper_nodal", "resistive_nodal", "gradient_nodal"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--battery",
        choices=sorted(BATTERIES),
        default=None,
        help="run a single battery in this process",
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default=None,
        help="run a set of batteries as subprocess children (each battery "
        "needs its own solver setup, and WarpX initializes once per process)",
    )
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    if args.suite is not None:
        # scrub the MPI launcher environment so each child runs as its own
        # singleton rather than trying to join the parent's MPI job
        import subprocess

        launcher_prefixes = ("PMI_", "PMIX_", "HYDRA_", "HYDI_", "OMPI_", "PRTE_")
        env = {
            k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)
        }
        for battery in SUITES[args.suite]:
            cmd = [sys.executable, os.path.abspath(__file__), "--battery", battery]
            result = subprocess.run(cmd, env=env, check=False)
            assert result.returncode == 0, f"{battery} battery failed"
            print(f"[suite {args.suite}] {battery} battery passed")
        return

    battery = args.battery or "hyper"
    grid_type, runner = BATTERIES[battery]
    sim = setup_simulation(battery, grid_type=grid_type)
    runner(sim)


if __name__ == "__main__":
    main()
