#!/usr/bin/env python3
"""2D hybrid magnetic-diffusion finite/damping smoke for a parser-frozen,
spatially varying (J-dependent) eta with theta < 1.

Not a Fourier oracle: a J-dependent eta has no single exponential decay mode.
Checks that By (component 0 of the plotfile Cell_H min/max blocks) is finite and
damps over the run. Mirrors the parsing in analysis.py but drops the discrete
amplification comparison.
"""

import glob
import math
import os
import re
import sys


def by_minmax(diag_dir):
    """Return (min, max) of By (component 0) from Level_0/Cell_H."""
    cell_h = os.path.join(diag_dir, "Level_0", "Cell_H")
    if not os.path.isfile(cell_h):
        raise FileNotFoundError(cell_h)
    with open(cell_h) as fh:
        text = fh.read()
    float_lines = []
    for line in text.splitlines():
        if re.match(r"^-?[0-9]", line.strip()):
            parts = [p for p in line.strip().rstrip(",").split(",") if p]
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) >= 3:
                float_lines.append(vals)
    if len(float_lines) < 2:
        raise RuntimeError(f"Could not parse By min/max from {cell_h}:\n{text}")
    # mins then maxs
    mins, maxs = float_lines[-2], float_lines[-1]
    return mins[0], maxs[0]


def main():
    plots = sorted(glob.glob("diags/diag1*"))
    if len(plots) < 2:
        print("FAILED: expected initial and final plotfiles")
        return 1

    bmin0, bmax0 = by_minmax(plots[0])
    bmin1, bmax1 = by_minmax(plots[-1])
    amp0 = max(abs(bmin0), abs(bmax0))
    amp1 = max(abs(bmin1), abs(bmax1))

    print(f"first={plots[0]} last={plots[-1]}")
    print(f"By amp t=0   = {amp0:.8e} T")
    print(f"By amp t_end = {amp1:.8e} T")

    if not all(math.isfinite(v) for v in (bmin0, bmax0, bmin1, bmax1)):
        print("FAILED: non-finite By value")
        return 1
    if amp0 < 1.0e-8:
        print("FAILED: initial By amplitude too small (B init missing?)")
        return 1
    if amp1 >= 0.999 * amp0:
        print("FAILED: By did not damp under variable-eta Crank-Nicolson diffusion")
        return 1

    print("PASSED: 2D hybrid magnetic-diffusion variable-eta CN finite/damping smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
