#!/usr/bin/env python3
"""Cylindrical hybrid magnetic diffusion with a Dirichlet B_t feed.

The upper-r pec_insulator feed drives toroidal B_t toward a cylindrical
profile with B_t ~ B0 at the feed face and small B_t near the axis
(B0 = 1.0e-4 T from insulator.By_x_hi after the ramp).

Checks (not a Fourier analytic solution):
  - feed face held near B0
  - near-axis B_t remains small
  - run remains finite/stable
"""

import glob
import math
import os
import re
import sys

B0 = 1.0e-4  # saturated feed value [T] (insulator.By_x_hi for t >= 1 ns)


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
    """Global (min, max) of a component across all fabs.

    WarpX plotfile Cell_H stores, after the grid header, two blocks: a per-fab
    MINIMUM block and a per-fab MAXIMUM block (each prefaced by a ``nfabs,ncomps``
    line). We take the elementwise min over the min-block and max over the
    max-block (not just the last two rows, which would both be maxima).
    """
    cell_h = os.path.join(diag_dir, "Level_0", "Cell_H")
    if not os.path.isfile(cell_h):
        raise FileNotFoundError(cell_h)
    with open(cell_h) as fh:
        text = fh.read()

    # Split into sections delimited by a lone "nfabs,ncomps" line.
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
        print("FAILED: expected initial and final cylindrical plotfiles")
        return 1

    # component 1 = Bt (toroidal) in cylindrical plot output (Br, Bt, Bz).
    bt_min0, bt_max0 = component_minmax(plotfiles[0], 1)
    bt_min1, bt_max1 = component_minmax(plotfiles[-1], 1)

    print(f"first={plotfiles[0]} last={plotfiles[-1]}")
    print(f"Bt t=0   min={bt_min0:.8e}  max={bt_max0:.8e} T")
    print(f"Bt t=end min={bt_min1:.8e}  max={bt_max1:.8e} T")

    if not all(math.isfinite(v) for v in (bt_min0, bt_max0, bt_min1, bt_max1)):
        print("FAILED: non-finite Bt value")
        return 1

    amp0 = max(abs(bt_min0), abs(bt_max0))
    amp1 = max(abs(bt_min1), abs(bt_max1))

    # Feed face (upper r) held near B0; axis (r=0) stays ~0 by regularity.
    if not (0.5 * B0 <= bt_max1 <= 1.5 * B0):
        print(f"FAILED: feed face Bt not held near B0 (max_end={bt_max1:.3e})")
        return 1
    if abs(bt_min1) > 0.05 * B0:
        print(f"FAILED: axis Bt not near 0 (min_end={bt_min1:.3e})")
        return 1
    # Clear radial profile: feed-face amplitude >> axis amplitude.
    if bt_max1 <= 10.0 * max(abs(bt_min1), 1.0e-12):
        print("FAILED: no clear radial profile (feed face not >> axis)")
        return 1
    # The feed must inject field: amplitude grew from the zero initialization.
    if amp1 <= 1.0001 * amp0 and amp0 > 0.0:
        print("FAILED: Bt amplitude did not grow under the feed")
        return 1

    print("PASSED: cylindrical hybrid magnetic-diffusion Dirichlet B_t feed smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
