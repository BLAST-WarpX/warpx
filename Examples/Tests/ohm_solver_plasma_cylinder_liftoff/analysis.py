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
BZ_BIAS = -0.1

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
