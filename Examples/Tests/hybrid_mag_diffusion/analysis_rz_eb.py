#!/usr/bin/env python3
"""RZ hybrid magnetic-diffusion smoke WITH a stair-case embedded boundary.

Oracle (not analytic): with an outer-wall EB (solid for r > r_wall, fluid on
axis) and mild parser eta = 1e-2 Ohm*m, theta = 1:
  - Bt min/max finite (no NaN/Inf)
  - initial Bt amplitude nonzero (uniform 1e-4 init applied)
  - final fluid Bt nonzero (max > threshold; the field persists in the fluid)
  - final Bt strictly damped vs initial
  - covered-region Bt inert: global min ~ 0 (solid faces carry Bt = 0; Bt >= 0
    everywhere so the global min is the covered value)

Uses the global Cell_H min/max parser (not analysis_rz.py's per-fab one, which
misreads gradient fields like this solid-vs-fluid Bt).
"""

import glob
import math
import os
import re
import sys


def _float_rows(text):
    """Yield the per-fab [Br, Bt, Bz, ...] float rows in a Cell_H block."""
    rows = []
    for line in text.splitlines():
        if re.match(r"^-?[0-9]", line.strip()):
            try:
                values = [
                    float(v) for v in line.strip().rstrip(",").split(",") if v.strip()
                ]
            except ValueError:
                continue
            if len(values) >= 3:
                rows.append(values)
    return rows


def component_minmax(diag_dir, component):
    """Global (min, max) of a component across all fabs in an RZ plotfile."""
    cell_h = os.path.join(diag_dir, "Level_0", "Cell_H")
    if not os.path.isfile(cell_h):
        raise FileNotFoundError(cell_h)
    with open(cell_h) as infile:
        text = infile.read()
    # Cell_H stores two float blocks after the grid header: per-fab min rows,
    # then per-fab max rows (each prefaced by a lone "nfabs,ncomps" line).
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

    # component 1 = Bt (toroidal) in RZ plot output (Br, Bt, Bz).
    bt_min0, bt_max0 = component_minmax(plotfiles[0], 1)
    bt_min1, bt_max1 = component_minmax(plotfiles[-1], 1)
    amp0 = max(abs(bt_min0), abs(bt_max0))
    amp1 = max(abs(bt_min1), abs(bt_max1))

    print(f"first={plotfiles[0]} last={plotfiles[-1]}")
    print(f"Bt t=0   min={bt_min0:.8e}  max={bt_max0:.8e} T")
    print(f"Bt t=end min={bt_min1:.8e}  max={bt_max1:.8e} T")

    if not all(math.isfinite(v) for v in (bt_min0, bt_max0, bt_min1, bt_max1)):
        print("FAILED: non-finite Bt value")
        return 1
    if amp0 < 1.0e-8:
        print("FAILED: initial Bt was not initialized")
        return 1
    if amp1 < 1.0e-8:
        print("FAILED: final fluid Bt annihilated (should persist at mild eta)")
        return 1
    if amp1 >= 0.999 * amp0:
        print("FAILED: Bt did not damp")
        return 1
    # Covered-region Bt inert: the solid (r > r_wall) carries Bt = 0, and Bt >= 0
    # everywhere, so the global min must be ~0 (not a large negative blow-up).
    if bt_min1 < -1.0e-8:
        print(f"FAILED: covered-region Bt not inert (min={bt_min1:.8e})")
        return 1

    print(
        "PASSED: RZ hybrid magnetic-diffusion EB smoke (covered Bt inert, fluid finite)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
