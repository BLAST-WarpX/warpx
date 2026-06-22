#!/usr/bin/env python3
#
# --- Cylinder near-wall (edge) order diagnostic (NOT registered with CTest).
# --- Measures the order of accuracy of the hybrid-PIC embedded-boundary
# --- magnetic-field treatment on a SMOOTH circular wall (no corners, unlike
# --- the rotated square in eb_diffusion) via the INTERIOR L2 error of By
# --- against the exact decaying Bessel eigenmode
# ---     By = B1 J0(J11 r/R) exp(-gamma t),  gamma = (eta/mu0)(J11/R)^2,
# --- sampled a fixed physical margin inside the wall so the same region is
# --- compared at every resolution. The interior solution is set by the
# --- embedded-boundary treatment near the wall, so its convergence is the
# --- edge order (this is the metric that gave the staggered ECT ~2.8 on the
# --- square; the wall tangential-J residual was physical-J-contaminated and
# --- blind to the clean's fluid-band action).
# ---
# --- Compares, at doubling resolutions:
# ---   - conformal ECT (staggered)             : the 2nd-order reference,
# ---   - mirror (collocated), clean OFF         : the bare level-set mirror,
# ---   - mirror (collocated), EB-aware clean ON : mirror + the fluid-stencil
# ---       div(B)/div(J) clean. Matching clean-OFF proves the clean preserves
# ---       the edge order. (On the ohms_law_conformal_nodal_ect branch, add a
# ---       collocated-C-ECT config to test whether C-ECT recovers 2nd order.)
# ---
# --- Example:  python3 cylinder_edge_order.py -N 48 96

import argparse
import glob
import math
import os
import subprocess
import sys

import numpy as np
import yt
from scipy.constants import mu_0
from scipy.special import j0

yt.set_log_level("error")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(THIS_DIR, "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py")
R_CYL = 0.6
J11 = 3.8317059702075125
B1 = 0.01
ETA = 1.0e-3
GAMMA = ETA / mu_0 * (J11 / R_CYL) ** 2  # cylinder Bessel-mode decay rate
MARGIN = 0.15  # interior region r < R_CYL - MARGIN (fixed physical, like square)
MAX_STEPS = 200
SQ_DECAY = ETA / mu_0 * (np.pi / 1.06) ** 2  # the deck's dt is set by the square rate

# (label, conformal, grid_type, divb_clean, eb_cyl)
CONFIGS = [
    ("conformal ECT (staggered)", True, "staggered", False, False),
    ("conformal ECT (staggered) + cyl-correction", True, "staggered", False, True),
    ("mirror (collocated), clean OFF", True, "collocated", False, False),
    ("mirror (collocated) + cyl-correction", True, "collocated", False, True),
]


def auto_substeps(resolution):
    diffusivity = ETA / mu_0
    dt = 0.5 / SQ_DECAY / MAX_STEPS
    dx = 1.6 / resolution
    dy = 0.05
    lam = 4 * diffusivity * (2.0 / dx**2 + 1.0 / dy**2)
    return max(4, 2 * math.ceil(lam * dt / 2.0))


def interior_err(plotfile):
    """Relative interior L2 error of By vs the decaying Bessel eigenmode."""
    ds = yt.load(plotfile)
    g = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    t = float(ds.current_time)
    lo = ds.domain_left_edge.to_ndarray()
    hi = ds.domain_right_edge.to_ndarray()
    n = np.array(ds.domain_dimensions)
    dcell = (hi - lo) / n
    x = lo[0] + (np.arange(n[0]) + 0.5) * dcell[0]
    z = lo[2] + (np.arange(n[2]) + 0.5) * dcell[2]
    xx, zz = np.meshgrid(x, z, indexing="ij")
    r = np.sqrt(xx * xx + zz * zz)
    by_th = B1 * j0(J11 * r / R_CYL) * np.exp(-GAMMA * t)
    by_sim = np.mean(g["By"].to_ndarray(), axis=1)
    interior = r < (R_CYL - MARGIN)
    err = np.sqrt(
        np.sum((by_sim - by_th)[interior] ** 2) / np.sum(by_th[interior] ** 2)
    )
    return float(err), float(dcell[0])


def run_case(n, conformal, grid_type, divb_clean, eb_cyl, outdir):
    tag = (
        f"{grid_type}_{'conf' if conformal else 'stair'}"
        f"{'_dc' if divb_clean else ''}{'_cyl' if eb_cyl else ''}_N{n}"
    )
    rundir = os.path.join(outdir, tag)
    os.makedirs(rundir, exist_ok=True)
    cmd = [
        sys.executable,
        INPUTS,
        "--geometry",
        "cylinder",
        "-n",
        str(n),
        "--substeps",
        str(auto_substeps(n)),
        "--grid-type",
        grid_type,
    ]
    if conformal:
        cmd += ["--conformal"]
    if divb_clean:
        cmd += ["--divb-clean"]
    if eb_cyl:
        cmd += ["--eb-cyl-correction"]
    print(f"  running {tag} ...")
    with open(os.path.join(rundir, "run.log"), "w") as log:
        subprocess.run(
            cmd, cwd=rundir, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    pf = sorted(glob.glob(os.path.join(rundir, "diags", "diag1" + "[0-9]" * 6)))[-1]
    return interior_err(pf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", "--resolutions", type=int, nargs="+", default=[48, 96])
    ap.add_argument("-o", "--outdir", default="cyl_edge_runs")
    args = ap.parse_args()

    print(f"\ncylinder edge-order (interior L2 vs Bessel mode): N={args.resolutions}\n")
    for label, conf, gt, dc, ec in CONFIGS:
        print(f"[{label}]")
        errs, hs = [], []
        for n in args.resolutions:
            e, h = run_case(n, conf, gt, dc, ec, args.outdir)
            errs.append(e)
            hs.append(h)
            print(f"    N={n}: interior L2 = {e:.4e}")
        o = float(np.log(errs[0] / errs[-1]) / np.log(hs[0] / hs[-1]))
        print(f"  edge order (By interior L2) = {o:.2f}")


if __name__ == "__main__":
    main()
