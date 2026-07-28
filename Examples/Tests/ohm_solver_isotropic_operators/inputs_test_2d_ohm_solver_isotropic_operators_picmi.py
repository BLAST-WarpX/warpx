#!/usr/bin/env python3
#
# --- 2D XZ variant of the isotropized Ohm's-law operator batteries: the
# --- Mehrstellen 9-point hyper-resistivity Laplacian, the corner-curl
# --- isotropization of the resistive diffusion of the out-of-plane By
# --- (carried by Ex and Ez), and the transverse-smoothed electron-pressure
# --- gradient, on both the staggered (Yee) and collocated groupings. See the
# --- 3D battery for the methodology; the assertions are the same
# --- (closed-form round-off match, teeth vs the standard stencil, low-degree
# --- exactness, and >= 4x suppression of the axis/diagonal cos(4*theta)
# --- anisotropy at kh = 1).

import argparse
import os
import sys

import numpy as np

from pywarpx import fields, picmi

constants = picmi.constants

N = 32
LO = -1.0
HI = 1.0
H = (HI - LO) / N  # square cells

ETA_H = 1.0
ETA_R = 1.0
RHO0 = 100.0 * 1.0e16 * constants.q_e


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
    # species-free single-box deck (the stencils must not straddle box seams)
    hyper = battery.startswith("hyper")
    resistive = battery.startswith("resistive")

    grid = picmi.Cartesian2DGrid(
        number_of_cells=[N, N],
        lower_bound=[LO, LO],
        upper_bound=[HI, HI],
        lower_boundary_conditions=["dirichlet", "periodic"],
        upper_boundary_conditions=["dirichlet", "periodic"],
        lower_boundary_conditions_particles=["absorbing", "periodic"],
        upper_boundary_conditions_particles=["absorbing", "periodic"],
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
        plasma_resistivity=(ETA_R if resistive else 0.0),
        plasma_hyper_resistivity=(ETA_H if hyper else None),
        # every battery runs with the isotropized operators enabled; the
        # other terms are inert through their zero coefficients above
        isotropic_operators=True,
        substeps=4,
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


# numpy mirrors (axis 0 = x with dirichlet edges, axis 1 = z periodic;
# asserts stay on interior slices so the wrap rows never enter a read)
def d2(a, ax):
    return np.roll(a, -1, ax) - 2.0 * a + np.roll(a, 1, ax)


def lap_cross(a):
    return (d2(a, 0) + d2(a, 1)) / H**2


def lap_iso(a):
    # Mehrstellen 9-point: cross + corner/(6 h^2)
    return lap_cross(a) + d2(d2(a, 0), 1) / (6.0 * H**2)


# ----------------------------------------------------------------------------
# Hyper-resistivity battery
# ----------------------------------------------------------------------------
def run_hyper_battery(sim, nodal=False):
    wx = sim.extension.warpx
    ck = CheckSet()
    zero_all_inputs()

    # Jy is the out-of-plane current: fully cc on Yee XZ, nodal collocated --
    # either way its own neighbours carry the compact stencils
    Jpy = fields.JyFPPlasmaWrapper()
    Ey = fields.EyFPWrapper()
    shape = np.asarray(Jpy[...]).shape

    interior = (slice(4, 29), slice(2, shape[1] - 2))
    tag = "nodal hyper" if nodal else "hyper"

    rng = np.random.RandomState(2024)
    jy = rng.rand(*shape) - 0.5
    Jpy[...] = jy
    wx.hybrid_solve_e(True)
    ey = np.asarray(Ey[...])
    expected = -ETA_H * lap_iso(jy)
    scale = float(np.max(np.abs(expected[interior])))
    ck.close(
        f"{tag}: solver matches the Mehrstellen 9-point closed form",
        ey[interior] / scale,
        expected[interior] / scale,
        1e-12,
    )
    d_cross = float(
        np.max(np.abs(ey[interior] + ETA_H * lap_cross(jy)[interior])) / scale
    )
    ck.expect(
        f"{tag}: result differs from the cross stencil (teeth)",
        d_cross > 1e-3,
        f"rel dev from cross = {d_cross:.3e}",
    )

    # plane-wave damping anisotropy, axis vs diagonal at kh = 1
    xs = np.arange(shape[0]) * H
    zs = np.arange(shape[1]) * H
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    kmag = 1.0 / H

    def solver_rate(jf):
        Jpy[...] = jf
        wx.hybrid_solve_e(True)
        lam = (np.asarray(Ey[...]) / (ETA_H * jf))[interior]
        return float(np.median(lam[np.abs(jf[interior]) > 0.5]))

    def stencil_rate(lap, jf):
        lam = (-lap(jf) / jf)[interior]
        return float(np.median(lam[np.abs(jf[interior]) > 0.5]))

    ja = np.cos(kmag * X)
    jd = np.cos(kmag * (X + Z) / np.sqrt(2.0))
    aniso_iso = abs(solver_rate(ja) - solver_rate(jd)) / kmag**2
    aniso_cross = (
        abs(stencil_rate(lap_cross, ja) - stencil_rate(lap_cross, jd)) / kmag**2
    )
    ck.expect(
        f"{tag}: axis/diagonal damping anisotropy suppressed (>= 4x at kh=1)",
        aniso_iso < 0.25 * aniso_cross,
        f"aniso(iso)/aniso(cross) = {aniso_iso / max(aniso_cross, 1e-300):.4f}",
    )

    ck.finish()


# ----------------------------------------------------------------------------
# Resistive corner-curl battery: out-of-plane By, correction carried by Ex
# and Ez; the Faraday curl (dBy/dt = UpwardDx(Ez) - UpwardDz(Ex) on Yee, wide
# centered on collocated) must emit the Mehrstellen corner.
# ----------------------------------------------------------------------------
def run_resistive_battery(sim, nodal=False):
    wx = sim.extension.warpx
    ck = CheckSet()
    mu0 = constants.mu0
    zero_all_inputs()

    By = fields.ByFPWrapper()
    shp = np.asarray(By[...]).shape
    tag = "nodal resistive" if nodal else "resistive"

    NC = (N, N)  # common crop (Ex, Ez, By have different staggered shapes)

    def crop(a):
        return np.asarray(a)[: NC[0], : NC[1]]

    def curl_y(ex, ez):
        # the Faraday increment of the out-of-plane By from E
        if nodal:
            dz_ex = (np.roll(ex, -1, 1) - np.roll(ex, 1, 1)) / (2.0 * H)
            dx_ez = (np.roll(ez, -1, 0) - np.roll(ez, 1, 0)) / (2.0 * H)
        else:
            dz_ex = (np.roll(ex, -1, 1) - ex) / H
            dx_ez = (np.roll(ez, -1, 0) - ez) / H
        return dz_ex - dx_ez

    ii = (slice(5, 26), slice(5, 26))

    def Dn(F, ax):
        return (np.roll(F, -1, ax) - np.roll(F, 1, ax)) / (2.0 * H)

    def corner_fn(a):
        # the extra operator the correction emits through the Faraday curl:
        # Yee (compact one-sided pre-differences completing centered second
        # differences) vs nodal (wide centered Faraday differences)
        if nodal:
            return (1.0 / 3.0) * (Dn(Dn(d2(a, 0), 1), 1) + Dn(Dn(d2(a, 1), 0), 0))
        return d2(d2(a, 0), 1) / (6.0 * H * H)

    rng = np.random.RandomState(7)
    by = rng.rand(*shp) - 0.5
    By[...] = by
    wx.hybrid_solve_e(True)
    Ex = crop(fields.ExFPWrapper()[...])
    Ez = crop(fields.EzFPWrapper()[...])
    by_c = crop(by)

    # emergent operator: the corner measured through the Faraday curl,
    # dBy/dt = -(curl E)_y = (eta/mu0) * (corner) for the correction-only E
    corner = corner_fn(by_c)
    curl_meas = curl_y(Ex, Ez)
    scale = float(np.max(np.abs(corner[ii]))) * (ETA_R / mu0)
    ck.close(
        f"{tag}: Faraday curl of the correction is the in-plane corner",
        -curl_meas[ii] / scale,
        (ETA_R / mu0) * corner[ii] / scale,
        1e-9,
    )
    ck.expect(
        f"{tag}: correction is nonzero (teeth)",
        float(np.max(np.abs(curl_meas[ii]))) > 1e-3 * scale,
        f"max|curl| = {float(np.max(np.abs(curl_meas[ii]))):.3e}",
    )

    # anisotropy of the emergent operator (base cross + corner) vs the plain
    # base cross alone; the base is the diffusion the plain resistive term
    # produces through the Faraday curl (compact on Yee, wide on collocated)
    xs = np.arange(NC[0]) * H
    zs = np.arange(NC[1]) * H
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    kmag = 1.0 / H

    def base_cross(a):
        if nodal:
            return Dn(Dn(a, 0), 0) + Dn(Dn(a, 1), 1)
        return lap_cross(a)

    def emergent(a):
        return base_cross(a) + corner_fn(a)

    def rate(lap_fn, prof):
        lam = (-lap_fn(prof) / prof)[ii]
        return float(np.median(lam[np.abs(prof[ii]) > 0.5]))

    ja = np.cos(kmag * X)
    jd = np.cos(kmag * (X + Z) / np.sqrt(2.0))
    a_cross = abs(rate(base_cross, ja) - rate(base_cross, jd)) / kmag**2
    a_iso = abs(rate(emergent, ja) - rate(emergent, jd)) / kmag**2
    ck.expect(
        f"{tag}: axis/diagonal damping anisotropy suppressed (>= 4x at kh=1)",
        a_iso < 0.25 * a_cross,
        f"aniso(iso)/aniso(cross)={a_iso / max(a_cross, 1e-300):.4f}",
    )

    # pure truncation canceller: zero on quadratics
    By[...] = np.broadcast_to((np.arange(shp[0]) * H)[:, None] ** 2, shp).copy()
    wx.hybrid_solve_e(True)
    ex = np.asarray(fields.ExFPWrapper()[...])
    ck.close(
        f"{tag}: correction vanishes on quadratics",
        ex[5:26, 5:26],
        0.0,
        1e-9 * float(np.max(np.abs(ex))) + 1e-12,
    )

    ck.finish()


# ----------------------------------------------------------------------------
# Gradient battery: Ex carries d/dx (transverse z), Ez carries d/dz
# (transverse x)
# ----------------------------------------------------------------------------
def run_gradient_battery(sim, nodal=False):
    wx = sim.extension.warpx
    ck = CheckSet()
    zero_all_inputs()

    Pe = fields.ElectronPressureFPWrapper()
    pe_shape = np.asarray(Pe[...]).shape  # nodal (N+1, N+1)
    tag = "nodal gradient" if nodal else "gradient"

    A_T = (1.0 / 6.0) if nodal else (1.0 / 24.0)

    def smooth(a, ax):
        return a + A_T * d2(a, ax)

    if nodal:

        def grad_x(a):
            return (np.roll(a, -1, 0) - np.roll(a, 1, 0)) / (2.0 * H)

    else:

        def grad_x(a):
            return (a[1:, :] - a[:-1, :]) / H

    def grad_x_iso(a):
        return grad_x(smooth(a, 1))

    nx_out = pe_shape[0] if nodal else pe_shape[0] - 1
    ii = (slice(3, nx_out - 3), slice(2, pe_shape[1] - 2))

    def solve_grad_x():
        wx.hybrid_solve_e(False)
        return -RHO0 * np.asarray(fields.ExFPWrapper()[...])

    # closed form on a random field + teeth
    rng = np.random.RandomState(11)
    pe = rng.rand(*pe_shape) - 0.5
    Pe[...] = pe
    gx = solve_grad_x()
    scale = float(np.max(np.abs(grad_x_iso(pe)[ii])))
    ck.close(
        f"{tag}: solver matches the transverse-smoothed closed form",
        gx[ii] / scale,
        grad_x_iso(pe)[ii] / scale,
        1e-12,
    )
    d_plain = float(np.max(np.abs(gx[ii] - grad_x(pe)[ii])) / scale)
    ck.expect(
        f"{tag}: result differs from the plain stencil (teeth)",
        d_plain > 1e-3,
        f"rel dev = {d_plain:.3e}",
    )

    # axis/diagonal symbol anisotropy at kh = 1
    xn = LO + np.arange(pe_shape[0]) * H
    zn = np.arange(pe_shape[1]) * H
    Xg, Zg = np.meshgrid(xn, zn, indexing="ij")
    if nodal:
        Xf, Zf = Xg, Zg
    else:
        xf = LO + (np.arange(pe_shape[0] - 1) + 0.5) * H
        Xf, Zf = np.meshgrid(xf, zn, indexing="ij")
    kmag = 1.0 / H

    def symbol_x(kx, kz, grad_fn=None):
        pe_w = np.sin(kx * Xg + kz * Zg)
        if grad_fn is None:
            Pe[...] = pe_w
            g = solve_grad_x()
        else:
            g = grad_fn(pe_w)
        cosf = np.cos(kx * Xf + kz * Zf)
        ratio = (g / (kx * cosf))[ii]
        return float(np.median(ratio[np.abs(cosf[ii]) > 0.5]))

    err_axis = symbol_x(kmag, 0.0) - 1.0
    err_diag = symbol_x(kmag / np.sqrt(2.0), kmag / np.sqrt(2.0)) - 1.0
    err_axis_p = symbol_x(kmag, 0.0, grad_x) - 1.0
    err_diag_p = symbol_x(kmag / np.sqrt(2.0), kmag / np.sqrt(2.0), grad_x) - 1.0
    aniso_iso = abs(err_axis - err_diag)
    aniso_plain = abs(err_axis_p - err_diag_p)
    ck.expect(
        f"{tag}: axis error unchanged (isotropization equalizes, not improves)",
        abs(err_axis - err_axis_p) <= 1e-10,
        f"axis rel err iso = {err_axis:.5f}, plain = {err_axis_p:.5f}",
    )
    ck.expect(
        f"{tag}: axis/diagonal symbol anisotropy suppressed (>= 4x at kh=1)",
        aniso_iso < 0.25 * aniso_plain,
        f"aniso(iso)/aniso(plain) = {aniso_iso / max(aniso_plain, 1e-300):.4f} "
        f"(plain = {aniso_plain:.5f}, iso = {aniso_iso:.6f})",
    )

    ck.finish()


BATTERIES = {
    "hyper": ("staggered", lambda sim: run_hyper_battery(sim, nodal=False)),
    "resistive": ("staggered", lambda sim: run_resistive_battery(sim, nodal=False)),
    "gradient": ("staggered", lambda sim: run_gradient_battery(sim, nodal=False)),
    "hyper_nodal": ("collocated", lambda sim: run_hyper_battery(sim, nodal=True)),
    "resistive_nodal": (
        "collocated",
        lambda sim: run_resistive_battery(sim, nodal=True),
    ),
    "gradient_nodal": (
        "collocated",
        lambda sim: run_gradient_battery(sim, nodal=True),
    ),
}

SUITES = {
    "staggered": ("hyper", "resistive", "gradient"),
    "collocated": ("hyper_nodal", "resistive_nodal", "gradient_nodal"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery", choices=sorted(BATTERIES), default=None)
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
