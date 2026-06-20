#!/usr/bin/env python3
#
# --- Local near-wall (edge) order diagnostic (NOT registered with CTest).
# --- Measures the order of accuracy of the hybrid-PIC embedded-boundary
# --- magnetic-field treatment AT THE WALL for three treatments:
# ---   - stair (staggered)         : the stair-step approximation,
# ---   - conformal ECT (staggered) : the enlarged-cell-technique B push,
# ---   - mirror (collocated)       : the nodal level-set mirror B fill.
# ---
# --- Reuses the eb_diffusion resistive-decay eigenmode and its analysis:
# ---   - compute_errors    -> interior L2 order of By and J (= E/eta),
# ---   - wall_bc_residuals -> the EDGE metric: tangential J at the wall (the
# ---       PEC condition drives it to 0) and the normal-J normal gradient
# ---       (Neumann). Their decrease with resolution is the edge order.
# ---
# --- The point of the conformal-EB PR is to raise the near-wall order; this
# --- driver quantifies it and sets up the mirror-vs-C-ECT A/B (the collocated
# --- C-ECT lives on the ohms_law_conformal_nodal_ect branch -- run this same
# --- driver there with a C-ECT config added to compare).
# ---
# --- Example:
# ---   python3 edge_order.py -N 32 64 --np 1

import argparse
import glob
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import DECAY_RATE, compute_errors, wall_bc_residuals  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(THIS_DIR, "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py")
MAX_STEPS = 200

# (label, conformal, grid_type)
CONFIGS = [
    ("stair (staggered)", False, "staggered"),
    ("conformal ECT (staggered)", True, "staggered"),
    ("mirror (collocated)", True, "collocated"),
]


def auto_substeps(resolution):
    """Even substep count keeping the RK stage step inside the explicit
    diffusion stability limit (mirrors plot_convergence.py)."""
    from scipy.constants import mu_0

    eta = 1.0e-3
    diffusivity = eta / mu_0
    dt = 0.5 / DECAY_RATE / MAX_STEPS
    dx = 1.6 / resolution
    dy = 0.05
    lam = 4 * diffusivity * (2.0 / dx**2 + 1.0 / dy**2)
    substeps = 2 * math.ceil(lam * dt / 1.0 / 2.0)
    return max(4, substeps)


def run_case(resolution, conformal, grid_type, nprocs, outdir):
    tag = f"{grid_type}_{'conf' if conformal else 'stair'}_N{resolution}"
    rundir = os.path.join(outdir, tag)
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
        "--grid-type",
        grid_type,
    ]
    if conformal:
        cmd += ["--conformal"]
    print(f"  running {tag} ...")
    with open(os.path.join(rundir, "run.log"), "w") as log:
        subprocess.run(
            cmd, cwd=rundir, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    plotfile = sorted(glob.glob(os.path.join(rundir, "diags", "diag1" + "[0-9]" * 6)))[
        -1
    ]
    err_b, err_j, h = compute_errors(plotfile)
    jt, djn = wall_bc_residuals(plotfile)
    return dict(h=h, err_b=err_b, err_j=err_j, jt=jt, djn=djn)


def order(lo, hi, h_lo, h_hi):
    return float(np.log(lo / hi) / np.log(h_lo / h_hi))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--resolutions", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--np", type=int, default=1, help="MPI ranks")
    parser.add_argument("-o", "--outdir", default="edge_order_runs")
    args = parser.parse_args()

    print(f"\nedge-order diagnostic: resolutions {args.resolutions}\n")
    rows = []
    for label, conformal, grid_type in CONFIGS:
        print(f"[{label}]")
        res = [
            run_case(n, conformal, grid_type, args.np, args.outdir)
            for n in args.resolutions
        ]
        lo, hi = res[0], res[-1]
        oB = order(lo["err_b"], hi["err_b"], lo["h"], hi["h"])
        oJ = order(lo["err_j"], hi["err_j"], lo["h"], hi["h"])
        # edge: the BC-residual convergence order (tangential J and normal-J
        # gradient both must decrease with resolution at the EB surface)
        oJt = order(lo["jt"], hi["jt"], lo["h"], hi["h"])
        oDjn = order(lo["djn"], hi["djn"], lo["h"], hi["h"])
        rows.append((label, oB, oJ, oJt, oDjn, hi))
        print(
            f"  interior order: B={oB:.2f} J={oJ:.2f} | "
            f"edge order: Jt={oJt:.2f} dJn={oDjn:.2f}  "
            f"(fine N={args.resolutions[-1]}: errB={hi['err_b']:.2e}, "
            f"wall|Jt|={hi['jt']:.2e})"
        )

    print("\n=== edge-order summary (order of accuracy) ===")
    print(f"{'treatment':<28}{'B int':>7}{'J int':>7}{'Jt edge':>9}{'dJn edge':>10}")
    for label, oB, oJ, oJt, oDjn, _ in rows:
        print(f"{label:<28}{oB:>7.2f}{oJ:>7.2f}{oJt:>9.2f}{oDjn:>10.2f}")
    print(
        "\n(reference: staggered conformal ECT ~2.8 interior; stair ~1.0-1.3; "
        "the planar mirror is curvature-limited so expect sub-2nd order.)"
    )


if __name__ == "__main__":
    main()
