#!/usr/bin/env python3
#
# --- Analysis for the theta-pinch liftoff slab: verifies that all fields are
# --- finite, that the annular column is loaded with the bias field threaded
# --- through it, and that the configuration is azimuthally uniform at the
# --- particle-noise level (the Cartesian grid seeds numerical modes, most
# --- prominently m=4, whose growth this case exists to monitor).
# ---
# --- The same script doubles as the azimuthal-mode probe for long liftoff
# --- runs: point --path at any plotfile to print the m-spectra of rho and Bz
# --- on the annulus mid-radius circle (use --radius to follow the imploding
# --- column).

import argparse

import numpy as np
import yt

# Defaults matching the simulation script
R_INNER = 0.6
R_OUTER = 0.7
R_WALL = 0.8
BZ_BIAS = -0.1
BZ_REV = 1.5
TAU_RAMP = 4.0e-6
N_I = 1.5e20
N_FLOOR_FRAC = 0.03
Q_E = 1.602176634e-19

# Azimuthal nonuniformity tolerances at the early-time CI checkpoint
# (calibrated at resolution 32 with 4 ppc: the rho spectrum sits at the
# particle-noise floor, modes m=1..8 at or below 0.012 of m=0; the external
# bias Bz is uniform to round-off)
TOL_RHO_MODES = 0.05
TOL_BZ_MODES = 0.01

N_THETA = 256
N_MODES = 8


def ring_spectrum(f, r0, lo, h):
    """Azimuthal amplitude spectrum |c_m| of the z-averaged cell-centered
    field f, bilinearly sampled on the circle of radius r0."""
    n = f.shape[0]
    th = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    px, py = r0 * np.cos(th), r0 * np.sin(th)
    fi = (px - lo) / h - 0.5
    fj = (py - lo) / h - 0.5
    i0 = np.clip(np.floor(fi).astype(int), 0, n - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, n - 2)
    wx, wy = fi - i0, fj - j0
    v = (
        (1 - wx) * (1 - wy) * f[i0, j0]
        + wx * (1 - wy) * f[i0 + 1, j0]
        + (1 - wx) * wy * f[i0, j0 + 1]
        + wx * wy * f[i0 + 1, j0 + 1]
    )
    return np.abs(np.fft.rfft(v)) / N_THETA


def bz_external(t):
    """Hermite-smoothstep external Bz ramp of the simulation script."""
    s = np.clip(t / TAU_RAMP, 0.0, 1.0)
    return BZ_BIAS + (BZ_REV - BZ_BIAS) * s * s * (3.0 - 2.0 * s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="plotfile to analyze")
    parser.add_argument(
        "--radius",
        type=float,
        default=0.5 * (R_INNER + R_OUTER),
        help="radius of the probe circle (default: annulus mid-radius)",
    )
    parser.add_argument(
        "--no-assert",
        action="store_true",
        help="report the mode spectra only (for probing long runs)",
    )
    parser.add_argument(
        "--marder",
        action="store_true",
        help="additionally probe plasma-flux conservation and the band "
        "divergence error (for runs with the transitional Marder cleaning)",
    )
    args = parser.parse_args()

    ds = yt.load(args.path)
    ad = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    lo = float(ds.domain_left_edge[0])
    n = int(ds.domain_dimensions[0])
    h = (float(ds.domain_right_edge[0]) - lo) / n

    fields = {
        name: np.asarray(ad[("boxlib", name)])
        for name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez", "rho", "jx", "jy", "jz")
    }
    for name, arr in fields.items():
        assert np.all(np.isfinite(arr)), f"non-finite values in {name}"

    # z-averaged in-plane maps (thin periodic slab)
    rho = fields["rho"].mean(axis=2)
    bz = fields["Bz"].mean(axis=2)

    # the annular column is loaded and the bias field threads the domain
    xc = lo + (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    R = np.sqrt(X**2 + Y**2)
    annulus = (R > R_INNER) & (R < R_OUTER)
    rho_annulus = rho[annulus].mean()
    print(f"mean rho in the annulus: {rho_annulus:.4e} C/m^3")

    # azimuthal mode spectra on the probe circle
    print(f"azimuthal spectra at r = {args.radius:.3f} m (relative to m=0):")
    rel = {}
    for name, f in (("rho", rho), ("Bz", bz)):
        c = ring_spectrum(f, args.radius, lo, h)
        rel[name] = c[1 : N_MODES + 1] / max(abs(c[0]), 1e-300)
        modes = "  ".join(f"m{m + 1}={a:.4f}" for m, a in enumerate(rel[name]))
        print(f"  {name:3s}: m0 = {c[0]:+.4e}  {modes}")

    if args.marder:
        # 1) plasma-flux conservation: the self-consistent axial flux inside
        # the conducting wall (total Bz minus the uniform external ramp at
        # this time) must stay near its initial value of zero - the Marder
        # band cleaning must not pump net flux (port of the Phoenix
        # marder-plus-RK45 flux-conservation test)
        t_now = float(ds.current_time)
        inside = R < R_WALL
        phi_plasma = float(np.sum((bz[inside] - bz_external(t_now)))) * h * h
        phi_scale = np.pi * R_WALL**2 * abs(BZ_REV)
        print(
            f"plasma axial flux inside the wall: {phi_plasma:.4e} Wb "
            f"({abs(phi_plasma) / phi_scale:.2e} of pi*R_w^2*|B_rev|)"
        )

        # 2) band divergence error: rms cell-centered div(E) in the
        # transition band (0 < rho <= n_floor*q_e) vs the plasma bulk - the
        # cleaned band must not be noisier than the bulk
        ex3, ey3, ez3 = fields["Ex"], fields["Ey"], fields["Ez"]
        dive = (
            np.gradient(ex3, h, axis=0)
            + np.gradient(ey3, h, axis=1)
            + np.gradient(ez3, h, axis=2)
        )
        rho3 = fields["rho"]
        rho_floor = N_FLOOR_FRAC * N_I * Q_E
        band3 = (rho3 > 0.0) & (rho3 <= rho_floor) & (R[:, :, None] < R_WALL - 2 * h)
        bulk3 = (rho3 > rho_floor) & (R[:, :, None] < R_WALL - 2 * h)
        rms_band = float(np.sqrt(np.mean(dive[band3] ** 2))) if band3.any() else 0.0
        rms_bulk = float(np.sqrt(np.mean(dive[bulk3] ** 2))) if bulk3.any() else 1.0
        ratio = rms_band / max(rms_bulk, 1e-300)
        print(f"div(E) rms: band={rms_band:.4e} bulk={rms_bulk:.4e} ratio={ratio:.3f}")

        if not args.no_assert:
            assert abs(phi_plasma) < 1.0e-3 * phi_scale, (
                f"plasma axial flux {phi_plasma:.3e} Wb exceeds 1e-3 of "
                f"pi*R_w^2*|B_rev| - the Marder cleaning is pumping flux"
            )
            assert ratio < 1.0, (
                f"band div(E) rms is {ratio:.2f}x the bulk - the Marder "
                "cleaning is not suppressing the band divergence noise"
            )

    if args.no_assert:
        return

    assert rho_annulus > 0.0, "no plasma loaded in the annulus"
    bz_axis = bz[R < 0.3].mean()
    assert abs(bz_axis - BZ_BIAS) < 0.05 * abs(BZ_BIAS), (
        f"on-axis Bz {bz_axis:.4f} T does not match the bias field {BZ_BIAS} T"
    )
    assert np.max(rel["rho"]) < TOL_RHO_MODES, (
        f"azimuthal density modes {np.max(rel['rho']):.3f} above the "
        f"particle-noise tolerance {TOL_RHO_MODES} at the early-time checkpoint"
    )
    assert np.max(rel["Bz"]) < TOL_BZ_MODES, (
        f"azimuthal Bz modes {np.max(rel['Bz']):.3f} above {TOL_BZ_MODES} "
        "at the early-time checkpoint"
    )


if __name__ == "__main__":
    main()
