#!/usr/bin/env python3
#
# Part B diagnostic (NOT registered with CTest): localize where the hybrid
# Ohm's-law embedded-boundary path loses convergence order on a smoothly
# curved wall.
#
# The conformal ECT magnetic-field advance is ~2nd order on a curved PEC wall
# (the EM-Maxwell oracle em_circle_cavity.py / em_circle_order.py), yet the
# hybrid resistive diffusion on the IDENTICAL cylinder + Bessel mode converges
# at only ~1st order (cylinder_edge_order.py). The B push is exonerated, so the
# 1st order must enter the hybrid near-wall E path: the Ampere curl curl(B)->J,
# the level-set mirror fill, and/or the algebraic Ohm assembly.
#
# This probe feeds the EXACT analytic eigenmode By = B1 J0(J11 r/R) into that
# path WITHOUT any time stepping and measures the per-component, radial-shell
# L2 order of
#   * the Ampere/plasma current  J = curl(B)/mu0     (hybrid_current_fp_plasma)
#   * the Ohm electric field      E = eta * J         (Efield_fp, holmstrom vac)
# against their continuum values. Two operations make it a no-time-integration
# measurement (so the result is the spatial operator order alone, not the B
# scheme's one-step error):
#   warpx.hybrid_calculate_plasma_current()  -> curl(B)/mu0 - J_ext + EB fill
#   warpx.hybrid_solve_e(True)               -> the algebraic Ohm E   + EB fill
#
# The initial By is overwritten with the analytic mode evaluated at each
# component's TRUE Yee-staggered location: the deck's afterinit hook places By
# at nodal coordinates, an O(h) approximation that would otherwise contaminate
# the operator-order measurement (curl of a misplaced field is 1st order
# everywhere). The continuum reference for J is
#   J = curl(B)/mu0,  B = B1 J0(J11 r/R) y_hat  ->
#   Jx =  (B1/mu0)(J11/R) J1(J11 r/R) (z/r),
#   Jz = -(B1/mu0)(J11/R) J1(J11 r/R) (x/r),  Jy = 0,  and E = eta J.
#
# A deep-interior order ~2 is the built-in sanity check that the staggered
# coordinates / analytic forms are correct; a near-wall order drop to ~1 then
# localizes the 1st-order entry point and feeds the Part C decision.
#
# Usage (driver):
#   ampere_curl_order.py [-N 48 96 ...] [--grid-type staggered|collocated]
#                        [--conformal] [--workdir DIR]
# Internal (one resolution per subprocess; ParmParse is single-init):
#   ampere_curl_order.py --run-one -n N --grid-type ... [--conformal] --out F

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

LO = -0.8  # x, z domain lower bound (m); the cylinder grid is N x 8 x N over [-0.8,0.8]^2 x [-0.2,0.2]
R_CYL = 0.6  # wall radius (overwritten from the dump metadata in the driver)


def axis_coords(size, ncell, lo, h):
    """Physical coordinates of a valid-cell array axis, inferring the Yee
    staggering from the size: ncell -> cell-centered, ncell+1 -> nodal."""
    if size == ncell:
        return lo + (np.arange(size) + 0.5) * h
    if size == ncell + 1:
        return lo + np.arange(size) * h
    raise ValueError(
        f"axis size {size} is neither {ncell} (cell) nor {ncell + 1} (nodal)"
    )


def xz_grid(shape, N):
    """(x, z) meshgrid at the staggered locations of a component whose valid-cell
    array has the given (nx, ny, nz) shape, on the N x 8 x N cylinder grid."""
    h = 1.6 / N
    xc = axis_coords(shape[0], N, LO, h)
    zc = axis_coords(shape[2], N, LO, h)
    return np.meshgrid(xc, zc, indexing="ij")  # (nx, nz)


def extrapolate_covered(by2d, XX, ZZ, R, h, band=2.0):
    """2nd-order, fluid-ONLY extension of B into the covered band via even reflection
    + local-quadratic interpolation. For each covered cell within `band` cells of the
    wall, reflect it RADIALLY across the wall to its image (radius 2R - r, INSIDE the
    fluid) and evaluate a distance-weighted local quadratic fit of the surrounding
    FLUID cells AT THE IMAGE. The axial By is tangential with dB/dn~0 at the wall, so
    the even reflection is 2nd-3rd order accurate; centering the fit at the (interior)
    image makes it INTERPOLATION, not the ill-conditioned one-sided extrapolation that
    overshoots. Differentiated by the curl this gives an O(h^2) near-wall current. The
    Part C prototype: in C++ it fits the discrete fluid B; here fluid B is the exact
    analytic mode, so this measures the SCHEME's order, not an analytic shortcut."""
    r = np.sqrt(XX * XX + ZZ * ZZ)
    fluid = r < R
    target = (r >= R) & (r < R + band * h)
    fx, fz, fv = XX[fluid], ZZ[fluid], by2d[fluid]
    out = by2d.copy()
    ii, kk = np.where(target)
    for i, k in zip(ii, kk):
        rc = np.hypot(XX[i, k], ZZ[i, k])
        scale = (2.0 * R - rc) / rc  # radial reflection: image at radius 2R - rc
        xim, zim = XX[i, k] * scale, ZZ[i, k] * scale
        # use a generous stencil so the 6-coef quadratic is well overdetermined
        for reach in (3.0 * h, 4.0 * h, 5.0 * h):
            sel = (np.abs(fx - xim) <= reach) & (np.abs(fz - zim) <= reach)
            if sel.sum() >= 12:
                break
        if sel.sum() < 6:
            continue  # too few fluid neighbors (rare); leave the prior value
        # non-dimensionalize by h so the quadratic columns are all O(1) (otherwise
        # dx^2 ~ h^2 makes lstsq ill-conditioned and WORSE as h->0); rcond truncates
        # the degenerate modes of a one-sided near-wall stencil so the fit degrades
        # gracefully to lower order instead of overshooting.
        u, v = (fx[sel] - xim) / h, (fz[sel] - zim) / h
        A = np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])
        coef, *_ = np.linalg.lstsq(A, fv[sel], rcond=1e-4)
        out[i, k] = coef[0]  # By at the image point (u=v=0), even parity
    return out


# ---------------------------------------------------------------------------
# Runner: build the sim, write the exact mode, evaluate the operators, dump.
# ---------------------------------------------------------------------------
def run_one(
    N,
    grid_type,
    conformal,
    out_path,
    mirror_b=False,
    continue_b=False,
    cyl_b=False,
    extrap_b=False,
    extrap_band=2.0,
    quad_gather=False,
    prod_bfill=False,
):
    sys.path.insert(0, HERE)
    from inputs_test_3d_ohm_solver_eb_diffusion_picmi import (
        B1,
        ETA,
        J11,
        R_CYL,
        setup_simulation,
    )
    from scipy.special import j0

    from pywarpx import callbacks, fields, libwarpx, picmi

    mu0 = picmi.constants.mu0
    k = J11 / R_CYL

    # exercise the PRODUCTION path: CalculatePlasmaCurrent itself extends the
    # covered B (quadratic) before the curl when conformal_b_curl_fill is set
    # (no --mirror-b). Pass it through the PICMI kwarg -- setting the raw bucket
    # attr is overwritten by HybridPICSolver.initialize_inputs.
    sim = setup_simulation(
        N,
        4,  # substeps (unused: the probe never advances the fields)
        conformal,  # use_conformal_eb
        False,  # pec_j
        0,  # verbose
        grid_type=grid_type,
        geometry="cylinder",
        b_curl_fill=prod_bfill,
    )

    def write_exact_by():
        """Overwrite By with B1 J0(J11 r/R) at By's true staggered coordinates
        (machine-exact analytic input for the curl), zero in the conductor."""
        Bw = fields.ByFPWrapper()
        arr = np.asarray(Bw[...])
        sh = arr.shape[:3]
        XX, ZZ = xz_grid(sh, N)
        r = np.sqrt(XX * XX + ZZ * ZZ)
        by2d = B1 * j0(k * r)
        if not continue_b:
            by2d[r > R_CYL] = 0.0  # staircased init (the deck's choice)
        # continue_b: leave the smooth analytic Bessel continuation in the
        # conductor so the curl reads an exact covered B (diagnostic ceiling)
        full = np.broadcast_to(by2d[:, None, :], sh)
        Bw[...] = full[..., None] if arr.ndim == 4 else np.ascontiguousarray(full)

    def extrap_covered_by():
        """Overwrite the covered band of By with the 2nd-order fluid-only quadratic
        extension (the Part C prototype fill the Ampere curl reads)."""
        Bw = fields.ByFPWrapper()
        arr = np.asarray(Bw[...])
        sh = arr.shape[:3]
        XX, ZZ = xz_grid(sh, N)
        by2d = arr.reshape(sh).mean(axis=1)  # y-invariant
        by2d = extrapolate_covered(by2d, XX, ZZ, R_CYL, 1.6 / N, band=extrap_band)
        full = np.broadcast_to(by2d[:, None, :], sh)
        Bw[...] = full[..., None] if arr.ndim == 4 else np.ascontiguousarray(full)

    def dump():
        write_exact_by()
        warpx = libwarpx.libwarpx_so.get_instance()
        if extrap_b:
            extrap_covered_by()  # 2nd-order fluid-only covered-B extension (Part C)
        if mirror_b:
            # A/B lever: mirror-fill the covered B (magnetic parity: normal odd,
            # tangential even) BEFORE the Ampere curl, so curl(B)->J reads an
            # extended covered B instead of the staircased init (covered=0). The
            # staggered path does NOT do this in production; this tests whether
            # it recovers the near-wall current order. fill_covered_centers=True
            # extends the covered-center faces (the cells the curl actually reads).
            # cyl_b: also apply the surface-of-revolution radial-Jacobian weighting
            # (axis y=1) that the J/rho fills use -- tests whether the cylindrical
            # mirror is 2nd-order-accurate enough to restore the near-wall curl.
            # quad_gather: use the 2nd-order even-reflection + quadratic fluid-only
            # gather (the C++ port of the validated --extrap-b prototype).
            warpx.hybrid_apply_eb_boundary_to_face_field(
                "Bfield_fp", 0, True, 1.0, cyl_b, 1, quad_gather
            )
        warpx.hybrid_calculate_plasma_current()  # curl(B)/mu0 - J_ext + EB fill
        warpx.hybrid_solve_e(True)  # algebraic Ohm E (= eta J in vacuum) + EB fill

        def rd(fn):
            return np.ascontiguousarray(np.squeeze(np.asarray(fn()[...])))

        np.savez(
            out_path,
            Jx=rd(fields.JxFPPlasmaWrapper),
            Jy=rd(fields.JyFPPlasmaWrapper),
            Jz=rd(fields.JzFPPlasmaWrapper),
            Ex=rd(fields.ExFPWrapper),
            Ey=rd(fields.EyFPWrapper),
            Ez=rd(fields.EzFPWrapper),
            Bx=rd(fields.BxFPWrapper),
            By=rd(fields.ByFPWrapper),
            Bz=rd(fields.BzFPWrapper),
            N=N,
            mu0=mu0,
            B1=B1,
            J11=J11,
            R_CYL=R_CYL,
            ETA=ETA,
        )
        print(f"[dump] wrote {out_path}", flush=True)

    callbacks.installafterinit(dump)
    sim.step(0)


# ---------------------------------------------------------------------------
# Driver: analytic reference, radial-shell L2 error, convergence order.
# ---------------------------------------------------------------------------
SHELLS = [
    (0.00, 0.45, "deep [0,.45]"),
    (0.45, 0.54, "mid  [.45,.54]"),
    (0.54, 0.60, "wall [.54,.60]"),
    (0.00, 0.60, "fluid[0,.60]"),
]
SIGNAL = ["Jx", "Jz", "Ex", "Ez"]
SANITY = ["By", "Jy"]


def analytic_2d(comp, shape, N, meta):
    """Continuum reference for a component, evaluated at its staggered (x,z)."""
    from scipy.special import j0, j1

    XX, ZZ = xz_grid(shape, N)
    r = np.sqrt(XX * XX + ZZ * ZZ)
    rsafe = np.where(r < 1e-30, 1.0, r)
    B1, mu0, J11, Rw, eta = (meta[key] for key in ("B1", "mu0", "J11", "R_CYL", "ETA"))
    k = J11 / Rw
    famp = (B1 / mu0) * k * j1(k * r)  # azimuthal current magnitude
    if comp == "Jx":
        a = famp * (ZZ / rsafe)
    elif comp == "Jz":
        a = -famp * (XX / rsafe)
    elif comp == "Ex":
        a = eta * famp * (ZZ / rsafe)
    elif comp == "Ez":
        a = -eta * famp * (XX / rsafe)
    elif comp == "By":
        a = B1 * j0(k * r)
    else:  # Jy, Ey, Bx, Bz
        a = np.zeros_like(r)
    return a, r


def shell_rms(sim2d, an2d, r, r0, r1, Rw):
    m = (r >= r0) & (r < r1) & (r < Rw)
    n = int(m.sum())
    if n == 0:
        return None
    err = float(np.sqrt(np.mean((sim2d[m] - an2d[m]) ** 2)))
    nrm = float(np.sqrt(np.mean(an2d[m] ** 2)))
    return err, nrm, n


def order(e1, e2, h1, h2):
    if e1 is None or e2 is None or e1 <= 0.0 or e2 <= 0.0:
        return float("nan")
    return float(np.log(e1 / e2) / np.log(h1 / h2))


def main_driver(
    resolutions,
    grid_type,
    conformal,
    workdir,
    mirror_b=False,
    continue_b=False,
    cyl_b=False,
    extrap_b=False,
    extrap_band=2.0,
    quad_gather=False,
    prod_bfill=False,
):
    os.makedirs(workdir, exist_ok=True)
    tag = (
        f"{grid_type}{'_conf' if conformal else ''}{'_mb' if mirror_b else ''}"
        f"{'_qg' if quad_gather else ''}{'_pb' if prod_bfill else ''}"
        f"{'_cyl' if cyl_b else ''}{'_ex' if extrap_b else ''}{'_cb' if continue_b else ''}"
    )

    dumps = {}
    for N in resolutions:
        rundir = os.path.join(workdir, f"n{N}_{tag}")
        os.makedirs(rundir, exist_ok=True)
        out = os.path.join(rundir, "dump.npz")
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--run-one",
            "-n",
            str(N),
            "--grid-type",
            grid_type,
            "--out",
            out,
        ]
        if conformal:
            cmd.append("--conformal")
        if mirror_b:
            cmd.append("--mirror-b")
        if continue_b:
            cmd.append("--continue-b")
        if cyl_b:
            cmd.append("--cyl-b")
        if extrap_b:
            cmd.append("--extrap-b")
            cmd += ["--extrap-band", str(extrap_band)]
        if quad_gather:
            cmd.append("--quad-gather")
        if prod_bfill:
            cmd.append("--prod-bfill")
        print(f"[run] N={N} ({tag}) ...", flush=True)
        with open(os.path.join(rundir, "log.txt"), "w") as log:
            subprocess.run(
                cmd, cwd=rundir, check=True, stdout=log, stderr=subprocess.STDOUT
            )
        if not os.path.exists(out):
            raise RuntimeError(
                f"runner produced no dump for N={N}; see {rundir}/log.txt"
            )
        dumps[N] = dict(np.load(out))

    Rw = float(dumps[resolutions[0]]["R_CYL"])
    hs = {N: 1.6 / N for N in resolutions}

    # Per-component, per-shell RMS at each resolution + order between consecutive Ns.
    print(
        f"\n=== ampere_curl_order: grid={grid_type} conformal={conformal} "
        f"mirror_b={mirror_b} quad_gather={quad_gather} cyl_b={cyl_b} "
        f"extrap_b={extrap_b} continue_b={continue_b} ==="
    )
    print(
        f"resolutions {resolutions}, h {[round(hs[N], 5) for N in resolutions]}, R_wall={Rw}"
    )
    print("(near-wall order drop from ~2 -> ~1 localizes the hybrid 1st-order entry)\n")

    for label, comps in (("SIGNAL", SIGNAL), ("SANITY (Jy,Ey~0; By=input)", SANITY)):
        print(f"--- {label} ---")
        header = f"{'comp':4} {'shell':16}" + "".join(
            f"{'err(' + str(N) + ')':>12}" for N in resolutions
        )
        for i in range(len(resolutions) - 1):
            header += (
                f"{'ord' + str(resolutions[i]) + '/' + str(resolutions[i + 1]):>10}"
            )
        print(header)
        for comp in comps:
            for r0, r1, sname in SHELLS:
                errs = []
                for N in resolutions:
                    arr = dumps[N][comp]
                    an, r = analytic_2d(comp, arr.shape, N, dumps[N])
                    sim2d = arr.mean(axis=1)  # fields are y-invariant
                    st = shell_rms(sim2d, an, r, r0, r1, Rw)
                    errs.append(None if st is None else st[0])
                row = f"{comp:4} {sname:16}" + "".join(
                    (f"{e:12.4e}" if e is not None else f"{'--':>12}") for e in errs
                )
                for i in range(len(resolutions) - 1):
                    row += f"{order(errs[i], errs[i + 1], hs[resolutions[i]], hs[resolutions[i + 1]]):10.2f}"
                print(row)
            print()
    # Relative near-wall magnitude context for the smallest h.
    Nf = resolutions[-1]
    print(f"--- near-wall relative L2 (N={Nf}) ---")
    for comp in SIGNAL:
        arr = dumps[Nf][comp]
        an, r = analytic_2d(comp, arr.shape, Nf, dumps[Nf])
        sim2d = arr.mean(axis=1)
        st = shell_rms(sim2d, an, r, 0.54, 0.60, Rw)
        if st is not None:
            err, nrm, n = st
            rel = err / nrm if nrm > 0 else float("nan")
            print(
                f"  {comp}: wall err {err:.3e}  ||analytic|| {nrm:.3e}  rel {rel:.3e}  (n={n})"
            )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("-n", "--resolution", type=int, help="single resolution (runner)")
    p.add_argument("--out", help="dump path (runner)")
    p.add_argument(
        "-N",
        "--resolutions",
        type=int,
        nargs="+",
        default=[48, 96],
        help="driver sweep",
    )
    p.add_argument(
        "--grid-type", choices=["staggered", "collocated"], default="staggered"
    )
    p.add_argument(
        "--conformal", action="store_true", help="use_conformal_eb (ECT on staggered)"
    )
    p.add_argument(
        "--mirror-b",
        action="store_true",
        help="A/B lever: mirror-fill covered B before the Ampere curl (the staggered "
        "path does not in production) -- tests if it recovers near-wall current order",
    )
    p.add_argument(
        "--continue-b",
        action="store_true",
        help="diagnostic ceiling: leave the smooth analytic Bessel continuation in the "
        "conductor (covered B exact) instead of the staircased covered=0 init",
    )
    p.add_argument(
        "--cyl-b",
        action="store_true",
        help="with --mirror-b, apply the surface-of-revolution radial-Jacobian weighting "
        "(axis y) to the covered-B mirror fill, like the J/rho fills",
    )
    p.add_argument(
        "--extrap-b",
        action="store_true",
        help="Part C prototype: fill covered B by a 2nd-order fluid-only quadratic "
        "extrapolation before the Ampere curl (should lift the near-wall current to order 2)",
    )
    p.add_argument(
        "--extrap-band",
        type=float,
        default=2.0,
        help="covered-B extrapolation band (cells)",
    )
    p.add_argument(
        "--quad-gather",
        action="store_true",
        help="with --mirror-b, use the C++ 2nd-order even-reflection + quadratic "
        "fluid-only covered-B gather (the port of the --extrap-b prototype)",
    )
    p.add_argument(
        "--prod-bfill",
        action="store_true",
        help="exercise the production wiring: set hybrid_pic_model.conformal_b_curl_fill "
        "so CalculatePlasmaCurrent extends covered B before the curl (no --mirror-b)",
    )
    p.add_argument("--workdir", default=os.path.join(HERE, "ampere_order_runs"))
    args = p.parse_args()

    if args.run_one:
        run_one(
            args.resolution,
            args.grid_type,
            args.conformal,
            args.out,
            args.mirror_b,
            args.continue_b,
            args.cyl_b,
            args.extrap_b,
            args.extrap_band,
            args.quad_gather,
            args.prod_bfill,
        )
    else:
        main_driver(
            args.resolutions,
            args.grid_type,
            args.conformal,
            args.workdir,
            args.mirror_b,
            args.continue_b,
            args.cyl_b,
            args.extrap_b,
            args.extrap_band,
            args.quad_gather,
            args.prod_bfill,
        )


if __name__ == "__main__":
    main()
