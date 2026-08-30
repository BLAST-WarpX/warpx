#!/usr/bin/env python3
"""RCYLINDER hybrid magnetic-diffusion PETSc parity oracle (T2 + T3).

Compares the PETSc assembled curl-curl run's final state to the AMReX GMRES
reference run (test_rcylinder_hybrid_mag_diffusion_mild, a ctest dependency whose
output directory survives until this test finishes). Same physics (mild eta,
PEC r_hi, theta=1, rtol=1e-10), differing only in the linear solver.

Strong solver-parity oracle:
  - reduced B-energy (FieldEnergy, column B_lev0) relative error < 1e-6
  - Bt Cell_H min/max relative error < 1e-6

The two Krylov drivers solve the SAME matrix-free operator (MATSHELL matvec =
linop.apply / computeAFull); with a well-posed PEC-r_hi system (the Bt~r null
mode excluded) they must converge to the same solution. Manual full-field L2
verification (yt) is ~1e-10; documented in
notes/2026-07-19_rcyl_mag_diff_port.md. This ctest uses the stdlib-readable
reduced B-energy + Cell_H min/max (the test python has no yt/numpy).
"""

import glob
import math
import os
import re
import sys

# Sibling ctest dependency directory (the AMReX GMRES reference run). The ctest
# `dependency` + cleanup ordering keep this directory alive through this test.
AMREX_REF_DIR = os.path.join("..", "test_rcylinder_hybrid_mag_diffusion_mild")
PARITY_TOL = 1.0e-6


def component_minmax(diag_dir, component):
    """Return the selected Cell_H component's (min, max)."""
    cell_h = os.path.join(diag_dir, "Level_0", "Cell_H")
    if not os.path.isfile(cell_h):
        raise FileNotFoundError(cell_h)
    with open(cell_h) as infile:
        text = infile.read()
    float_lines = []
    for line in text.splitlines():
        if re.match(r"^-?[0-9]", line.strip()):
            try:
                values = [
                    float(value)
                    for value in line.strip().rstrip(",").split(",")
                    if value
                ]
            except ValueError:
                continue
            if len(values) >= 3:
                float_lines.append(values)
    if len(float_lines) < 2:
        raise RuntimeError(f"Could not parse field min/max from {cell_h}")
    minimum, maximum = float_lines[-2], float_lines[-1]
    return minimum[component], maximum[component]


def last_plotfile(base_dir):
    plots = sorted(
        p
        for p in glob.glob(os.path.join(base_dir, "diags", "diag1*"))
        if os.path.isdir(p)
    )
    if not plots:
        raise RuntimeError(f"No plotfiles under {base_dir}/diags/")
    return plots[-1]


def b_energy(reduced_dir):
    """Return the last-row B_lev0(J) from a FieldEnergy reduced file."""
    path = os.path.join(reduced_dir, "field_energy.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path) as infile:
        lines = infile.readlines()
    # header line starts with '#'; data rows are "step time total E B"
    data = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    if not data:
        raise RuntimeError(f"No data rows in {path}")
    fields = data[-1].split()
    return float(fields[4])  # [4] B_lev0(J)


def relerr(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1.0e-30)


def main():
    amrex_plot = last_plotfile(AMREX_REF_DIR)
    petsc_plot = last_plotfile(".")
    amrex_energy = b_energy(os.path.join(AMREX_REF_DIR, "diags", "reduced_files"))
    petsc_energy = b_energy(os.path.join("diags", "reduced_files"))

    # RCYLINDER component order is Br, Bt, Bz; Bt is component 1.
    a_lo, a_hi = component_minmax(amrex_plot, 1)
    p_lo, p_hi = component_minmax(petsc_plot, 1)

    e_rel = relerr(amrex_energy, petsc_energy)
    lo_rel = relerr(a_lo, p_lo)
    hi_rel = relerr(a_hi, p_hi)

    print(f"AMReX ref: {amrex_plot}")
    print(f"PETSc    : {petsc_plot}")
    print(
        f"B-energy: AMReX={amrex_energy:.10e}  PETSc={petsc_energy:.10e}  "
        f"rel={e_rel:.3e}"
    )
    print(f"Bt min : AMReX={a_lo:.8e}  PETSc={p_lo:.8e}  rel={lo_rel:.3e}")
    print(f"Bt max : AMReX={a_hi:.8e}  PETSc={p_hi:.8e}  rel={hi_rel:.3e}")

    if not all(
        math.isfinite(v) for v in (amrex_energy, petsc_energy, a_lo, a_hi, p_lo, p_hi)
    ):
        print("FAILED: non-finite value")
        return 1
    if e_rel >= PARITY_TOL:
        print(f"FAILED: B-energy parity {e_rel:.3e} >= {PARITY_TOL:.0e}")
        return 1
    if lo_rel >= PARITY_TOL or hi_rel >= PARITY_TOL:
        print(
            f"FAILED: Bt min/max parity (min {lo_rel:.3e}, max {hi_rel:.3e}) "
            f">= {PARITY_TOL:.0e}"
        )
        return 1

    print(
        f"PASSED: RCYLINDER PETSc vs AMReX parity "
        f"(B-energy + Bt min/max rel < {PARITY_TOL:.0e})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
