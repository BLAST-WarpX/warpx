#!/usr/bin/env python3
#
# --- STEP 1 DISCRIMINATOR (a vs b): does WarpX's EM-Maxwell conformal (ECT)
# --- B-update reach 2nd order on a CURVED PEC wall, reproducing
# --- Dey-Mittra/Xiao-Liu? Identical geometry + Bessel J0 Neumann mode as the
# --- hybrid eb_diffusion cylinder test, but the EM WAVE operator instead of
# --- resistive diffusion. The mode TM^y_01 is By = B1 J0(J11 r/R) cos(omega t),
# --- omega = c J11 / R (Neumann dBy/dr = 0 at the wall = PEC tangential E = 0).
# --- The field-L2 error vs the analytic mode at a fixed time is dominated by the
# --- discrete-frequency (eigenvalue) error * t, so its convergence order is the
# --- eigenvalue order -- directly comparable to the hybrid's decay-rate order
# --- (measured ~1) and to the literature's 2nd-order frequency claim.
# ---   2nd order here  -> EM ECT works on curves; the hybrid 1st-order is the
# ---                      diffusive/Ohm-mirror COUPLING (case b).
# ---   1st order here  -> WarpX ECT curved-wall IMPLEMENTATION gap (case a).
#
# --- Single-run deck. Driver: em_circle_order.py.

import argparse

import numpy as np
from scipy.constants import c

from pywarpx import callbacks, fields, picmi

R_CYL = 0.6
J11 = 3.8317059702075125  # first zero of J1 = first nonzero extremum of J0 (Neumann)
B1 = 1.0e-3
OMEGA = c * J11 / R_CYL
PERIOD = 2.0 * np.pi / OMEGA
LO, HI = -0.8, 0.8


def init_bessel_by():
    """afterinit hook: By = B1 J0(J11 r/R), zero in the conductor; E = 0. The
    TM^y_01 standing mode (max B, zero E) -> By(t) = B1 J0(J11 r/R) cos(omega t).
    Written via the field wrapper because the parser has no Bessel."""
    from scipy.special import j0

    By = fields.ByFPWrapper()
    arr = np.asarray(By[...])
    nx, nz = arr.shape[0], arr.shape[-1]
    print(f"[init] By wrapper shape = {arr.shape}  (N grid cells per side)", flush=True)
    x = np.linspace(LO, HI, nx)
    z = np.linspace(LO, HI, nz)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    r = np.sqrt(xx * xx + zz * zz)
    by2d = B1 * j0(J11 * r / R_CYL)
    by2d[r > R_CYL] = 0.0
    By[...] = by2d.reshape(arr.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--solver", default="ECT", help="ECT or Yee")
    ap.add_argument("--periods", type=float, default=3.0)
    ap.add_argument("--cfl", type=float, default=0.95)
    args = ap.parse_args()
    N = args.n

    grid = picmi.Cartesian2DGrid(
        number_of_cells=[N, N],
        lower_bound=[LO, LO],
        upper_bound=[HI, HI],
        lower_boundary_conditions=["dirichlet", "dirichlet"],  # -> pec
        upper_boundary_conditions=["dirichlet", "dirichlet"],  # -> pec
        lower_boundary_conditions_particles=["absorbing", "absorbing"],
        upper_boundary_conditions_particles=["absorbing", "absorbing"],
        warpx_max_grid_size=2048,
        warpx_blocking_factor=8,
    )
    solver = picmi.ElectromagneticSolver(grid=grid, method=args.solver, cfl=args.cfl)

    sim = picmi.Simulation(solver=solver, verbose=0)
    sim.embedded_boundary = picmi.EmbeddedBoundary(
        implicit_function="sqrt(x*x+z*z)-rcyl", rcyl=R_CYL
    )

    dx = (HI - LO) / N
    dt = args.cfl * dx / (c * np.sqrt(2.0))
    nsteps = max(8, int(round(args.periods * PERIOD / dt)))

    diag = picmi.FieldDiagnostic(
        name="diag1",
        grid=grid,
        period=max(1, nsteps // 60),  # ~60 dumps through the run for an omega fit
        data_list=["B"],
        write_dir="diags",
        warpx_format="plotfile",
    )
    sim.add_diagnostic(diag)

    callbacks.installafterinit(init_bessel_by)
    sim.step(nsteps)


if __name__ == "__main__":
    main()
