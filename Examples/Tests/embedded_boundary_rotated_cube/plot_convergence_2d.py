#!/usr/bin/env python3
#
# --- Local convergence-study driver (not registered with CTest). Runs the 2D
# --- rotated-square PEC cavity at a series of resolutions with the stair-step
# --- Yee solver and with the conformal ECT (enlarged cell technique) solver,
# --- fits the order of accuracy of each, and writes a log-log convergence
# --- plot reproducing Fig. 3 of Xiao and Liu, IEEE Microwave Wireless
# --- Compon. Lett. 14, 551 (2004).
# ---
# --- Example:
# ---   python3 plot_convergence_2d.py --exe ../../../build_eb/bin/warpx.2d.* -N 32 64 128 256

import argparse
import glob
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_convergence_2d import CAVITY_SIDE, compute_error  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(THIS_DIR, "inputs_test_2d_embedded_boundary_rotated_cube")


def find_default_exe():
    repo = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
    candidates = sorted(glob.glob(os.path.join(repo, "build*", "bin", "warpx.2d*")))
    return candidates[-1] if candidates else None


def run_case(exe, solver, resolution, nprocs, outdir):
    rundir = os.path.join(outdir, f"{solver}_N{resolution}")
    os.makedirs(rundir, exist_ok=True)
    cmd = []
    if nprocs > 1:
        cmd += ["mpirun", "-np", str(nprocs)]
    cmd += [
        exe,
        INPUTS,
        f"algo.maxwell_solver={solver}",
        f"amr.n_cell={resolution} {resolution}",
    ]
    print(f"running: {' '.join(cmd)} (in {rundir})")
    with open(os.path.join(rundir, "run.log"), "w") as log:
        subprocess.run(
            cmd, cwd=rundir, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    plotfile = sorted(glob.glob(os.path.join(rundir, "diags", "diag1" + "[0-9]" * 6)))[
        -1
    ]
    return compute_error(plotfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", default=find_default_exe(), help="2D WarpX executable")
    parser.add_argument(
        "-N",
        "--resolutions",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="resolutions to run",
    )
    parser.add_argument("--np", type=int, default=1, help="number of MPI ranks")
    parser.add_argument(
        "-o", "--outdir", default="convergence_runs", help="scratch directory"
    )
    args = parser.parse_args()
    if not args.exe:
        parser.error("no 2D WarpX executable found, pass --exe")

    results = {}
    for solver in ("yee", "ect"):
        errs = []
        hs = []
        for n in args.resolutions:
            err, h = run_case(args.exe, solver, n, args.np, args.outdir)
            print(f"{solver} N={n}: err = {err:.4e}")
            errs.append(err)
            hs.append(h)
        order = np.polyfit(np.log(hs), np.log(errs), 1)[0]
        results[solver] = (np.array(hs), np.array(errs), order)
        print(f"{solver}: fitted order = {order:.2f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = {"yee": "stair-step Yee", "ect": "conformal ECT"}
    markers = {"yee": "x", "ect": "o"}
    for solver, (hs, errs, order) in results.items():
        ax.loglog(
            CAVITY_SIDE / hs,
            errs,
            markers[solver],
            ls="none",
            label=f"{labels[solver]} (order {order:.2f})",
        )
    ppw = CAVITY_SIDE / results["ect"][0]
    ax.loglog(
        ppw,
        results["ect"][1][0] * (ppw / ppw[0]) ** (-2.0),
        "k-",
        lw=0.8,
        label="2nd order",
    )
    ax.loglog(
        ppw,
        results["yee"][1][0] * (ppw / ppw[0]) ** (-1.0),
        "k--",
        lw=0.8,
        label="1st order",
    )
    ax.set_xlabel("cells per cavity side")
    ax.set_ylabel("relative interior L2 error of $B_y$")
    ax.set_title("EM solver, rotated PEC cavity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("convergence_em.png", dpi=200)
    print("wrote convergence_em.png")


if __name__ == "__main__":
    main()
