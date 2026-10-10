#!/usr/bin/env python3
"""Full-field RCYLINDER PETSc/AMReX solver parity."""

import os
import sys

from analysis_plotfile import (
    all_finite,
    last_plotfile,
    load_fields,
    relative_field_norms,
)

AMREX_REF_DIR = os.path.join("..", "test_rcylinder_hybrid_mag_diffusion_mild")
PARITY_TOL = 1.0e-6


def main():
    names = ("Br", "Bt", "Bz")
    _, reference = load_fields(last_plotfile(AMREX_REF_DIR), names)
    _, candidate = load_fields(last_plotfile(), names)

    errors = []
    for name in names:
        if not all_finite(reference[name]) or not all_finite(candidate[name]):
            print(f"FAILED: {name} contains NaN/Inf")
            return 1
        l2, linf = relative_field_norms(reference[name], candidate[name])
        errors.extend((l2, linf))
        print(f"{name}: relative L2={l2:.3e}, relative Linf={linf:.3e}")

    if max(errors) >= PARITY_TOL:
        print(f"FAILED: full-field RCYLINDER parity error >= {PARITY_TOL:.0e}")
        return 1
    print(f"PASSED: full-field RCYLINDER PETSc/AMReX parity < {PARITY_TOL:.0e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
