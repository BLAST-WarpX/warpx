#!/usr/bin/env python3
"""RZ hybrid magnetic diffusion with PEC endwalls on non-periodic z.

Both axial faces are PEC. A uniform Bt is tangential to the endwalls (mirrored);
Bz is the normal component and is pinned to zero.

Checks (finite/damping, not a Fourier analytic solution):
  - Bt is finite
  - initial |Bt| is nonzero
  - final peak |Bt| is strictly smaller than the initial peak (damping)
  - Bz remains approximately zero

Reads AMReX plotfile Cell_H min/max (no yt/openPMD).
"""

import glob
import math
import os
import re
import sys


def _float_rows(text):
    """Yield the per-fab [Br,Bt,Bz,...] float rows in a Cell_H block."""
    rows = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if re.match(r"^-?[0-9]", line):
            try:
                values = [float(v) for v in line.split(",") if v.strip()]
            except ValueError:
                continue
            if len(values) >= 3:
                rows.append(values)
    return rows


def component_minmax(diag_dir, component):
    """Global (min, max) of a component across all fabs (per-fab min/max blocks)."""
    cell_h = os.path.join(diag_dir, "Level_0", "Cell_H")
    if not os.path.isfile(cell_h):
        raise FileNotFoundError(cell_h)
    with open(cell_h) as fh:
        text = fh.read()

    sections = re.split(r"^\s*\d+,\s*\d+\s*$", text, flags=re.MULTILINE)
    float_sections = [s for s in (_float_rows(sec) for sec in sections) if s]
    if len(float_sections) < 2:
        raise RuntimeError(f"Could not parse min/max blocks from {cell_h}")

    min_rows = float_sections[-2]
    max_rows = float_sections[-1]
    gmin = min(r[component] for r in min_rows)
    gmax = max(r[component] for r in max_rows)
    return gmin, gmax


def main():
    plotfiles = sorted(glob.glob("diags/diag1*"))
    if len(plotfiles) < 2:
        print("FAILED: expected initial and final RZ plotfiles")
        return 1

    # RZ plot output order: (Br, Bt, Bz) -> indices 0, 1, 2.
    bt_min0, bt_max0 = component_minmax(plotfiles[0], 1)
    bt_min1, bt_max1 = component_minmax(plotfiles[-1], 1)
    bz_min1, bz_max1 = component_minmax(plotfiles[-1], 2)

    bt_amp0 = max(abs(bt_min0), abs(bt_max0))
    bt_amp1 = max(abs(bt_min1), abs(bt_max1))

    print(f"first={plotfiles[0]} last={plotfiles[-1]}")
    print(f"Bt range t=0   = [{bt_min0:.8e}, {bt_max0:.8e}] T")
    print(f"Bt range t=end = [{bt_min1:.8e}, {bt_max1:.8e}] T")
    print(f"Bz range t=end = [{bz_min1:.8e}, {bz_max1:.8e}] T")

    if not all(
        math.isfinite(v) for v in (bt_min0, bt_max0, bt_min1, bt_max1, bz_min1, bz_max1)
    ):
        print("FAILED: non-finite B value")
        return 1
    if bt_amp0 < 1.0e-8:
        print("FAILED: initial Bt was not initialized")
        return 1
    # Resistive damping: peak amplitude must drop (no Faraday/implicit blow-up).
    if bt_amp1 >= 0.999 * bt_amp0:
        print("FAILED: Bt did not damp in the RZ z-PEC eta=129 smoke")
        return 1
    # PEC zeros the z-normal component on the endwalls; with a Bt-only init the
    # bulk Bz stays ~0. Allow a loose floor for cell-tangential noise.
    bz_amp1 = max(abs(bz_min1), abs(bz_max1))
    if bz_amp1 > 1.0e-5:
        print(
            f"FAILED: Bz grew unexpectedly large ({bz_amp1:.3e}); "
            f"PEC endwall should keep B_normal ~ 0"
        )
        return 1

    print("PASSED: RZ hybrid magnetic-diffusion z-PEC endwall smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
