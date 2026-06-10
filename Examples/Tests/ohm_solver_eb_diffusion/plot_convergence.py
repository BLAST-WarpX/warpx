#!/usr/bin/env python3
#
# --- Local convergence-study driver (not registered with CTest). Runs the
# --- hybrid-PIC embedded-boundary diffusion test at a series of resolutions
# --- with both the stair-step and the conformal (enlarged cell technique)
# --- embedded-boundary treatments, fits the order of accuracy, and writes a
# --- log-log convergence plot in the spirit of Fig. 3 of Xiao and Liu,
# --- IEEE Microwave Wireless Compon. Lett. 14, 551 (2004).
# ---
# --- Example:
# ---   python3 plot_convergence.py -N 32 64 128 256 --np 2

import argparse
import glob
import math
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import CAVITY_SIDE, DECAY_RATE, compute_error  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(THIS_DIR, "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py")
MAX_STEPS = 200


def auto_substeps(resolution):
    """Even substep count keeping the RK4 stage step well inside the explicit
    diffusion stability limit, with margin for enlarged cut cells."""
    from scipy.constants import mu_0

    eta = 1.0e-3
    diffusivity = eta / mu_0
    dt = 0.5 / DECAY_RATE / MAX_STEPS
    dx = 1.6 / resolution
    dy = 0.05
    lam = 4 * diffusivity * (2.0 / dx**2 + 1.0 / dy**2)
    substeps = 2 * math.ceil(lam * dt / 1.0 / 2.0)
    return max(4, substeps)


def run_case(resolution, conformal, nprocs, outdir):
    rundir = os.path.join(
        outdir, f"{'conformal' if conformal else 'stair'}_N{resolution}"
    )
    os.makedirs(rundir, exist_ok=True)
    cmd = []
    if nprocs > 1:
        cmd += ["mpirun", "-np", str(nprocs)]
    cmd += [
        sys.executable,
        INPUTS,
        "-n",
        str(resolution),
        "--substeps",
        str(auto_substeps(resolution)),
    ]
    if conformal:
        cmd += ["--conformal"]
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
    parser.add_argument(
        "-N",
        "--resolutions",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="in-plane resolutions to run",
    )
    parser.add_argument("--np", type=int, default=2, help="number of MPI ranks")
    parser.add_argument(
        "-o", "--outdir", default="convergence_runs", help="scratch directory"
    )
    args = parser.parse_args()

    results = {}
    for mode, conformal in (("stair-step", False), ("conformal (ECT)", True)):
        errs = []
        hs = []
        for n in args.resolutions:
            err, h = run_case(n, conformal, args.np, args.outdir)
            print(f"{mode} N={n}: err = {err:.4e}")
            errs.append(err)
            hs.append(h)
        order = np.polyfit(np.log(hs), np.log(errs), 1)[0]
        results[mode] = (np.array(hs), np.array(errs), order)
        print(f"{mode}: fitted order = {order:.2f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    markers = {"stair-step": "x", "conformal (ECT)": "o"}
    for mode, (hs, errs, order) in results.items():
        ax.loglog(
            CAVITY_SIDE / hs,
            errs,
            markers[mode],
            ls="none",
            label=f"{mode} (order {order:.2f})",
        )
    ppw = CAVITY_SIDE / results["conformal (ECT)"][0]
    ref = results["conformal (ECT)"][1][0]
    ax.loglog(ppw, ref * (ppw / ppw[0]) ** (-2.0), "k-", lw=0.8, label="2nd order")
    ref1 = results["stair-step"][1][0]
    ax.loglog(ppw, ref1 * (ppw / ppw[0]) ** (-1.0), "k--", lw=0.8, label="1st order")
    ax.set_xlabel("cells per cavity side")
    ax.set_ylabel("relative interior L2 error of $B_y$")
    ax.set_title("Hybrid solver, EB resistive diffusion")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("convergence_hybrid.png", dpi=200)
    print("wrote convergence_hybrid.png")


if __name__ == "__main__":
    main()
