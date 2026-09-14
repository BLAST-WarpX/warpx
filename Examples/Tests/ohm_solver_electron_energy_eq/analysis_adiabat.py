#!/usr/bin/env python3
"""Validate the electron-energy-equation LHS (advection + compression) via
the adiabat.

For a sources-off, uniform-initial-entropy compression run, entropy-conserving
transport requires, at every cell and time,

    T_e(x,t) = T_e0 * ( n(x,t) / n0 )^(gamma - 1).

Low-density cells (below the n_floor used by the solver) are masked, since
T_e is gated there.

Produces:
  * left  : T_e(x) measured (solid) vs Te0 (n/n0)^(gamma-1) (dashed) at
            several times;
  * right : a scatter of T_e/Te0 vs n/n0 (all cells & times) that must
            collapse onto the single adiabat curve y = x^(gamma-1),
and checks the pointwise relative error against --tol-median / --tol-max.
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

Q_E = 1.602176634e-19
K_B = 1.380649e-23
K_PER_EV = K_B / Q_E


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--diag-dir", default="diags/field_diags")
    ap.add_argument("--gamma", type=float, default=5.0 / 3.0)
    ap.add_argument(
        "--n-floor-frac",
        type=float,
        default=0.05,
        help="mask cells with n < this fraction of n0 (matches the solver n_floor)",
    )
    ap.add_argument(
        "--tol-median",
        type=float,
        default=0.005,
        help="allowed median pointwise relative error on the adiabat",
    )
    ap.add_argument(
        "--tol-max",
        type=float,
        default=0.02,
        help="allowed max pointwise relative error on the adiabat",
    )
    ap.add_argument("--out", default="adiabat_check.png")
    args = ap.parse_args(argv)

    ts = OpenPMDTimeSeries(args.diag_dir)
    its = list(ts.iterations)
    times = np.asarray(ts.t, dtype=float)
    if len(its) < 2:
        raise SystemExit(f"Need >=2 dumps in {args.diag_dir}")
    g1 = args.gamma - 1.0

    def zavg(name, it):
        arr, info = ts.get_field(name, iteration=it)
        return np.asarray(info.x, dtype=float), np.asarray(arr, dtype=float).mean(
            axis=0
        )

    coord = None
    Te_x, n_x = [], []
    for it in its:
        cc, Te = zavg("Te", it)
        _, rho = zavg("rho", it)
        if coord is None:
            coord = cc
        Te_x.append(Te * K_PER_EV)
        n_x.append(rho / Q_E)
    Te_x = np.array(Te_x)  # (nt, nx)
    n_x = np.array(n_x)

    # Reference state = median of the first dump (uniform initial fill).
    Te0 = float(np.median(Te_x[0]))
    n0 = float(np.median(n_x[0]))
    Te_pred = Te0 * (n_x / n0) ** g1

    valid = n_x > args.n_floor_frac * n0
    rel = np.abs(Te_x - Te_pred) / np.maximum(Te_x, 1e-30)
    # Score only meaningfully compressed cells above the density floor.
    sig = valid & (np.abs(n_x / n0 - 1.0) > 0.03)
    med = float(np.median(rel[sig])) if np.any(sig) else float("nan")
    mx = float(np.max(rel[sig])) if np.any(sig) else float("nan")

    dn = float(np.max(np.abs((n_x / n0 - 1.0)[valid])))
    print("=" * 62)
    print("Adiabatic-compression check  Te = Te0 (n/n0)^(gamma-1)")
    print(f"  gamma = {args.gamma:.5f}   Te0 = {Te0:.2f} eV   n0 = {n0:.3e} m^-3")
    print(f"  peak density swing |n/n0 - 1| = {dn:.1%}  (cells above n_floor)")
    print(
        f"  relative error (compressed, above floor): median {med:.2%} "
        f"(tol {args.tol_median:.2%}), max {mx:.2%} (tol {args.tol_max:.2%})"
    )
    print("=" * 62)

    c_cm = coord * 100.0
    fig, (axP, axS) = plt.subplots(1, 2, figsize=(13, 5.0))

    nt = len(its)
    idxs = sorted(set(np.linspace(0, nt - 1, 5).astype(int)))
    for j in idxs:
        c = plt.cm.viridis(j / max(nt - 1, 1))
        axP.plot(
            c_cm,
            Te_x[j],
            "-",
            color=c,
            lw=1.8,
            label=f"t={times[j] * 1e6:.2f}" + r" $\mu$s",
        )
        axP.plot(c_cm, Te_pred[j], "--", color=c, lw=1.0)
    axP.set_xlabel("x (cm)")
    axP.set_ylabel("$T_e$ (eV)")
    axP.set_title("solid: measured $T_e$   dashed: $T_{e0}(n/n_0)^{\\gamma-1}$")
    axP.legend(fontsize=8, ncol=2)
    axP.grid(alpha=0.3)

    nn = (n_x / n0)[valid].ravel()
    tt = (Te_x / Te0)[valid].ravel()
    tcol = np.broadcast_to(times[:, None] * 1e6, n_x.shape)[valid].ravel()
    sc = axS.scatter(nn, tt, c=tcol, s=6, cmap="plasma", alpha=0.5)
    xs = np.linspace(nn.min(), nn.max(), 200)
    axS.plot(xs, xs**g1, "k-", lw=2, label=r"adiabat $(n/n_0)^{\gamma-1}$")
    fig.colorbar(sc, ax=axS, label=r"time ($\mu$s)")
    axS.set_xlabel("$n / n_0$")
    axS.set_ylabel("$T_e / T_{e0}$")
    axS.set_title(f"adiabat collapse  (median err {med:.2%}, max {mx:.2%})")
    axS.legend()
    axS.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[saved] {args.out}")

    ok = np.any(sig) and med <= args.tol_median and mx <= args.tol_max
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
