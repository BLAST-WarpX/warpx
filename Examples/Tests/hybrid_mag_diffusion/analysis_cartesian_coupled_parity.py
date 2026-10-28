#!/usr/bin/env python3
"""Compare every magnetic-field cell in two Cartesian solver runs."""

import os
import sys

from analysis_plotfile import (
    all_finite,
    last_plotfile,
    load_fields,
    relative_field_norms,
)

PARITY_TOL = 2.0e-6


def reference_dir():
    if len(sys.argv) > 1:
        return sys.argv[1]
    test_name = os.path.basename(os.getcwd())
    if not test_name.endswith("_petsc"):
        raise RuntimeError(f"Reference directory argument required for {test_name}")
    return os.path.join("..", test_name.removesuffix("_petsc"))


def main():
    names = ("Bx", "By", "Bz")
    _, reference = load_fields(last_plotfile(reference_dir()), names)
    _, candidate = load_fields(last_plotfile(), names)

    failed = False
    for name in names:
        if not all_finite(reference[name]) or not all_finite(candidate[name]):
            print(f"FAILED: {name} contains NaN/Inf")
            failed = True
            continue
        l2, linf = relative_field_norms(reference[name], candidate[name])
        print(f"{name}: relative L2={l2:.3e}, relative Linf={linf:.3e}")
        failed = failed or max(l2, linf) >= PARITY_TOL

    if failed:
        print(f"FAILED: full-field solver parity error >= {PARITY_TOL:.0e}")
        return 1
    print(f"PASSED: full-field solver parity < {PARITY_TOL:.0e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
