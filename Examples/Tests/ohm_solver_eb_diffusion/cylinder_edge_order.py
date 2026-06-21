#!/usr/bin/env python3
#
# --- Cylinder near-wall (edge) order diagnostic (NOT registered with CTest).
# --- Measures the near-wall order of accuracy of the hybrid-PIC embedded-
# --- boundary magnetic-field treatment on a SMOOTH circular wall (no corners,
# --- unlike the rotated square in eb_diffusion), via the wall-BC residual: the
# --- tangential current at the wall, which the PEC condition drives to zero.
# --- Its rate of decrease with resolution is the edge order. No analytic
# --- interior solution is needed (the cos(pi r^2/R^2) init is not an
# --- eigenmode), which is why the edge metric -- not an interior L2 -- is used.
# ---
# --- Compares, at doubling resolutions:
# ---   - conformal ECT (staggered)            : the 2nd-order reference,
# ---   - mirror (collocated), clean OFF        : the bare level-set mirror,
# ---   - mirror (collocated), EB-aware clean ON : mirror + the fluid-stencil
# ---       div(B)/div(J) clean. If the clean still preserved the edge order,
# ---       its order matches clean-OFF; the pre-fix centered clean dropped it.
# ---
# --- Example:  python3 cylinder_edge_order.py -N 48 96 --np 1

import argparse
import glob
import math
import os
import subprocess
import sys

import numpy as np
import yt
from scipy.constants import mu_0
from scipy.interpolate import RegularGridInterpolator

yt.set_log_level("error")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(THIS_DIR, "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py")
R_CYL = 0.6
MAX_STEPS = 200
DECAY_RATE = 1.0e-3 / mu_0 * (np.pi / 1.06) ** 2  # the square mode's rate (sets dt)

# (label, conformal, grid_type, divb_clean)
CONFIGS = [
    ("conformal ECT (staggered)", True, "staggered", False),
    ("mirror (collocated), clean OFF", True, "collocated", False),
    ("mirror (collocated), EB-aware clean ON", True, "collocated", True),
]


def auto_substeps(resolution):
    eta = 1.0e-3
    diffusivity = eta / mu_0
    dt = 0.5 / DECAY_RATE / MAX_STEPS
    dx = 1.6 / resolution
    dy = 0.05
    lam = 4 * diffusivity * (2.0 / dx**2 + 1.0 / dy**2)
    return max(4, 2 * math.ceil(lam * dt / 2.0))


def wall_tangential_j(plotfile):
    """Max tangential |J| at the cylinder wall r = R_CYL (the PEC condition
    drives it to zero), normalized by the peak current. Sampled a quarter cell
    on either side of the wall along the radius and averaged to the wall."""
    ds = yt.load(plotfile)
    g = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    lo = ds.domain_left_edge.to_ndarray()
    hi = ds.domain_right_edge.to_ndarray()
    n = np.array(ds.domain_dimensions)
    dcell = (hi - lo) / n
    h = dcell[0]
    x = lo[0] + (np.arange(n[0]) + 0.5) * dcell[0]
    z = lo[2] + (np.arange(n[2]) + 0.5) * dcell[2]
    # Ampere current = J + J_displacement, averaged over the thin y slab
    j = {
        c: np.mean(
            g[f"j{c}"].to_ndarray() + g[f"j{c}_displacement"].to_ndarray(), axis=1
        )
        for c in "xyz"
    }
    interp = {
        c: RegularGridInterpolator((x, z), j[c], bounds_error=False, fill_value=0.0)
        for c in "xyz"
    }
    scale = max(np.max(np.abs(j[c])) for c in "xyz")
    phis = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    samples = []
    for s in (-0.5 * h, 0.5 * h):
        r = R_CYL + s
        px, pz = r * np.cos(phis), r * np.sin(phis)
        fx = interp["x"]((px, pz))
        fy = interp["y"]((px, pz))
        fz = interp["z"]((px, pz))
        # tangential = in-plane azimuthal (phi_hat = (-sin, cos)) + axial (jy)
        ft = -fx * np.sin(phis) + fz * np.cos(phis)
        samples.append(np.sqrt(ft**2 + fy**2))
    jt_wall = 0.5 * (samples[0] + samples[1])
    return float(np.max(jt_wall) / scale)


def run_case(n, conformal, grid_type, divb_clean, outdir):
    tag = f"{grid_type}_{'conf' if conformal else 'stair'}{'_dc' if divb_clean else ''}_N{n}"
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
    print(f"  running {tag} ...")
    with open(os.path.join(rundir, "run.log"), "w") as log:
        subprocess.run(
            cmd, cwd=rundir, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    pf = sorted(glob.glob(os.path.join(rundir, "diags", "diag1" + "[0-9]" * 6)))[-1]
    return wall_tangential_j(pf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", "--resolutions", type=int, nargs="+", default=[48, 96])
    ap.add_argument("-o", "--outdir", default="cyl_edge_runs")
    args = ap.parse_args()

    print(f"\ncylinder edge-order (wall tangential-J residual): N={args.resolutions}\n")
    for label, conf, gt, dc in CONFIGS:
        print(f"[{label}]")
        jt = [run_case(n, conf, gt, dc, args.outdir) for n in args.resolutions]
        h = [1.6 / n for n in args.resolutions]
        o = float(np.log(jt[0] / jt[-1]) / np.log(h[0] / h[-1]))
        ratios = " -> ".join(f"{v:.3e}" for v in jt)
        print(f"  wall |Jt|/scale: {ratios}   edge order = {o:.2f}")


if __name__ == "__main__":
    main()
