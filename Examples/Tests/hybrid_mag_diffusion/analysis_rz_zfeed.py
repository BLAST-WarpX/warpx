#!/usr/bin/env python3
"""RZ hybrid magnetic diffusion with an axial pec_insulator B_t feed.

With z_lo feed B_t = B0*(r/R) and matching r_hi Dirichlet B_t = B0, the
cylindrical mode B_t ≈ B0*r/R is admitted. After several resistive steps the
field should fill the axial domain with max B_t near B0.

Checks (not a Fourier analytic solution):
  - peak B_t near the feed scale B0
  - deep axial fill of that profile
"""

import glob
import math
import os
import re
import sys

B0 = 1.0e-4  # saturated feed scale [T]


def _float_rows(text):
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
    cell_h = os.path.join(diag_dir, "Level_0", "Cell_H")
    if not os.path.isfile(cell_h):
        raise FileNotFoundError(cell_h)
    text = open(cell_h).read()
    sections = re.split(r"^\s*\d+,\s*\d+\s*$", text, flags=re.MULTILINE)
    float_sections = [s for s in (_float_rows(sec) for sec in sections) if s]
    if len(float_sections) < 2:
        raise RuntimeError(f"Could not parse min/max blocks from {cell_h}")
    min_rows = float_sections[-2]
    max_rows = float_sections[-1]
    gmin = min(r[component] for r in min_rows)
    gmax = max(r[component] for r in max_rows)
    return gmin, gmax


def bt_z_profile(diag_dir):
    """Return max_|Bt| vs z-index from the single-level fab (32 x 64)."""
    import struct

    path = os.path.join(diag_dir, "Level_0", "Cell_D_00000")
    with open(path, "rb") as f:
        f.readline()  # FAB header
        # 32 * 64 * 3 values, Fortran order (i, j, comp). Double or float
        # depending on WarpX_PRECISION.
        n = 32 * 64 * 3
        data = f.read()
    if len(data) == n * 8:
        vals = list(struct.unpack(f"{n}d", data))
    elif len(data) == n * 4:
        vals = list(struct.unpack(f"{n}f", data))
    else:
        raise RuntimeError(
            f"Unexpected Cell_D size {len(data)} (expected {n * 4} or {n * 8})"
        )
    # reshape (nr, nz, ncomp) Fortran; ncomp = 3 (Br, Bt, Bz)
    nr, nz = 32, 64
    bt_z = []
    for j in range(nz):
        m = 0.0
        for i in range(nr):
            # index: i + j*nr + 1*nr*nz for comp=1 (Bt)
            v = abs(vals[i + j * nr + 1 * nr * nz])
            if v > m:
                m = v
        bt_z.append(m)
    return bt_z


def main():
    plotfiles = sorted(glob.glob("diags/diag1*"))
    if len(plotfiles) < 2:
        print("FAILED: expected initial and final RZ plotfiles")
        return 1

    bt_min0, bt_max0 = component_minmax(plotfiles[0], 1)
    bt_min1, bt_max1 = component_minmax(plotfiles[-1], 1)

    print(f"first={plotfiles[0]} last={plotfiles[-1]}")
    print(f"Bt t=0   min={bt_min0:.8e}  max={bt_max0:.8e} T")
    print(f"Bt t=end min={bt_min1:.8e}  max={bt_max1:.8e} T")

    if not all(math.isfinite(v) for v in (bt_min0, bt_max0, bt_min1, bt_max1)):
        print("FAILED: non-finite Bt value")
        return 1

    # Start from zero
    if abs(bt_max0) > 1.0e-12:
        print(f"FAILED: expected near-zero init Bt, got max={bt_max0}")
        return 1

    # Feed has driven B_t up to nearly B0 (cylindrical null mode at r=R)
    if bt_max1 < 0.5 * B0:
        print(f"FAILED: final max Bt={bt_max1} < 0.5*B0 (feed not held)")
        return 1
    if bt_max1 > 1.5 * B0:
        print(f"FAILED: final max Bt={bt_max1} > 1.5*B0 (overshoot)")
        return 1

    # Deep axial fill: max_|Bt| at mid-z should be comparable to the endwall
    bt_z = bt_z_profile(plotfiles[-1])
    mid = bt_z[len(bt_z) // 2]
    lo = bt_z[0]
    print(
        f"max|Bt| z-lo={lo:.6e}  z-mid={mid:.6e}  ratio mid/lo={mid / max(lo, 1e-30):.4f}"
    )
    if mid < 0.5 * lo:
        print(
            "FAILED: mid-z Bt is less than half the endwall value — axial fill "
            "missing (would indicate a true axial operator bug, or an "
            "incompatible r_hi BC that forces a short cylindrical skin)"
        )
        return 1

    print(
        "PASSED: z-face pec_insulator B_t feed reaches deep axial fill "
        f"(max Bt={bt_max1:.4e} ~ B0, mid/lo={mid / max(lo, 1e-30):.3f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
