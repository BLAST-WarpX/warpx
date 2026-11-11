#!/usr/bin/env python3
"""Validate entropy-conserving transport through a below-floor halo.

The vacuum case drifts a plasma slab (n = n0) through a halo whose density is
below the solver's n_floor. The run starts from the floored adiabat (uniform
electron entropy K_e = T_e n_e^(1-gamma)), and has no sources (B = 0, eta = 0),
so entropy-conserving transport requires, at every cell and time,

    T_e(x,t) = T_e0 * ( n(x,t) / n0 )^(gamma - 1),

exactly -- a uniform K_e is invariant under any mass-weighted mixing, so the
check holds even through the CIC-mixed drifting slab edge. If the transport
instead left K_e = 0 in below-floor cells (an absorbing halo), the halo would
dilute and erase the slab's entropy at the edge: T_e would fall off the
adiabat within tens of steps and the slab's electron thermal energy would
drain away.

Scored on slab cells (n > 0.5 n0). Also reports the slab-mean T_e retention
between the first and last dump.
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
        "--slab-frac",
        type=float,
        default=0.5,
        help="score cells with n > this fraction of n0 (the slab interior)",
    )
    ap.add_argument(
        "--tol-median",
        type=float,
        default=0.01,
        help="allowed median pointwise relative error on the adiabat",
    )
    ap.add_argument(
        "--tol-max",
        type=float,
        default=0.02,
        help="allowed max pointwise relative error on the adiabat "
        "(correct transport stays below ~0.2%%; an absorbing K_e = 0 halo "
        "reaches ~8%% within the 80-step CI run)",
    )
    ap.add_argument("--out", default="vacuum_check.png")
    args = ap.parse_args(argv)

    ts = OpenPMDTimeSeries(args.diag_dir)
    # Skip the iteration-0 dump: it is written before the floored-adiabat
    # T_e seed runs (T_e is still the uniform InitData fill there).
    its = [it for it in ts.iterations if it > 0]
    times = np.asarray(ts.t, dtype=float)[-len(its) :]
    if len(its) < 2:
        raise SystemExit(f"Need >=2 post-step dumps in {args.diag_dir}")
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

    # Reference state from the first dump: the slab is the high-density mode.
    n0 = float(np.percentile(n_x[0], 90))
    slab0 = n_x[0] > args.slab_frac * n0
    Te0 = float(np.median(Te_x[0][slab0]))

    Te_pred = Te0 * (n_x / n0) ** g1
    slab = n_x > args.slab_frac * n0

    rel = np.abs(Te_x - Te_pred) / np.maximum(Te_pred, 1e-30)
    med = float(np.median(rel[slab]))
    mx = float(np.max(rel[slab]))

    # Slab-mean retention (density-weighted), reported for context only (not
    # asserted): edge rarefaction moves scored cells down the adiabat, so it
    # sits below 1 even for exact transport (~0.82 over the 80-step CI run),
    # and an absorbing halo shows up in the pointwise max error long before
    # it moves this mean.
    slab_end = n_x[-1] > args.slab_frac * n0
    Te_mean0 = float(np.sum((Te_x[0] * n_x[0])[slab0]) / np.sum(n_x[0][slab0]))
    Te_mean1 = float(np.sum((Te_x[-1] * n_x[-1])[slab_end]) / np.sum(n_x[-1][slab_end]))
    retention = Te_mean1 / Te_mean0

    print("=" * 62)
    print("Vacuum-slab (insulating halo) check  Te = Te0 (n/n0)^(gamma-1)")
    print(f"  gamma = {args.gamma:.5f}   Te0 = {Te0:.2f} eV   n0 = {n0:.3e} m^-3")
    print(f"  slab cells scored: n > {args.slab_frac:.2f} n0")
    print(
        f"  relative error on the adiabat: median {med:.2%} "
        f"(tol {args.tol_median:.2%}), max {mx:.2%} (tol {args.tol_max:.2%})"
    )
    print(f"  slab-mean Te retention (last/first dump): {retention:.4f}")
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

    nn = (n_x / n0)[slab].ravel()
    tt = (Te_x / Te0)[slab].ravel()
    tcol = np.broadcast_to(times[:, None] * 1e6, n_x.shape)[slab].ravel()
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

    ok = med <= args.tol_median and mx <= args.tol_max
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
