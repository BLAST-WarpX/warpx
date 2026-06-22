#!/usr/bin/env python3
#
# --- Driver for em_circle_cavity.py (NOT registered with CTest). Measures the
# --- resonant-frequency (eigenvalue) convergence order of the EM-Maxwell
# --- conformal (ECT) vs staircase (Yee) B-update on a SMOOTH circular PEC
# --- cavity. This validates that WarpX's EM ECT is ~2nd order on a curved wall
# --- (reproducing Dey-Mittra/Xiao-Liu) while Yee is 1st -- the oracle for the
# --- hybrid E-solve work (the hybrid is 1st order on the IDENTICAL geometry +
# --- J0 mode because its algebraic Ohm's-law E is not integrated by the
# --- EB-aware operator; see ECT_AWARE_ESOLVE_PLAN.md).
# ---
# --- The TM^y_01 standing mode By = B1 J0(J11 r/R) cos(omega t), omega = c J11/R,
# --- is projected onto J0 and the modal amplitude is fit to A cos(omega t + phi)
# --- to extract the discrete frequency. The field-L2-at-fixed-time metric is
# --- unreliable here (it sits near a phase node); use the frequency.
# --- NOTE: the t=0 diagnostic dump predates the afterinit field write (amplitude
# --- 0) and is dropped, else the cosine fit degenerates.
# ---
# --- Example:  python3 em_circle_order.py -N 64 128 256

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import yt
from scipy.constants import c
from scipy.optimize import curve_fit
from scipy.special import j0

yt.set_log_level("error")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(THIS_DIR, "em_circle_cavity.py")
R = 0.6
J11 = 3.8317059702075125
OMEGA = c * J11 / R


def freq_error(rundir):
    """Discrete resonant frequency from a cosine fit of the J0-modal amplitude."""
    pfs = sorted(glob.glob(os.path.join(rundir, "diags", "diag1" + "[0-9]" * 6)))[1:]
    ts, amps = [], []
    for pf in pfs:
        ds = yt.load(pf)
        g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        by = np.array(g["By"].to_ndarray()).squeeze()
        n = by.shape[0]
        ax = np.linspace(-0.8, 0.8, n)
        xx, zz = np.meshgrid(ax, ax, indexing="ij")
        r = np.sqrt(xx * xx + zz * zz)
        mode = j0(J11 * r / R)
        m = r < R
        ts.append(float(ds.current_time))
        amps.append(np.sum((by * mode)[m]) / np.sum((mode * mode)[m]))
    ts, amps = np.array(ts), np.array(amps)
    popt, _ = curve_fit(
        lambda t, a, w, ph: a * np.cos(w * t + ph),
        ts,
        amps,
        p0=[amps[0] or 1e-3, OMEGA, 0.0],
        maxfev=50000,
    )
    return abs(popt[1] - OMEGA) / OMEGA


def run_case(n, solver, periods, outdir):
    rundir = os.path.join(outdir, f"{solver}_N{n}")
    os.makedirs(rundir, exist_ok=True)
    cmd = [
        sys.executable,
        DECK,
        "-n",
        str(n),
        "--solver",
        solver,
        "--periods",
        str(periods),
    ]
    print(f"  running {solver} N={n} ...")
    with open(os.path.join(rundir, "run.log"), "w") as log:
        subprocess.run(
            cmd, cwd=rundir, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    return freq_error(rundir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", "--resolutions", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--periods", type=float, default=5.0)
    ap.add_argument("-o", "--outdir", default="em_circle_runs")
    args = ap.parse_args()

    print(
        f"\nEM circular-cavity frequency convergence (omega={OMEGA:.5e}): "
        f"N={args.resolutions}\n"
    )
    for solver in ("ECT", "Yee"):
        print(f"[{solver}]")
        errs = []
        for n in args.resolutions:
            e = run_case(n, solver, args.periods, args.outdir)
            errs.append(e)
            print(f"    N={n}: freq error = {e:.4e}")
        for i in range(1, len(errs)):
            o = np.log(errs[i - 1] / errs[i]) / np.log(
                args.resolutions[i] / args.resolutions[i - 1]
            )
            print(
                f"    order N{args.resolutions[i - 1]}->N{args.resolutions[i]} = {o:.2f}"
            )


if __name__ == "__main__":
    main()
