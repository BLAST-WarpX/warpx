#!/usr/bin/env python3
"""RZ constant-eta magnetic-diffusion vacuum stability smoke."""
import glob
import math
import os
import re
import sys


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
                values = [float(value) for value in line.strip().rstrip(",").split(",") if value]
            except ValueError:
                continue
            if len(values) >= 3:
                float_lines.append(values)

    if len(float_lines) < 2:
        raise RuntimeError(f"Could not parse field min/max from {cell_h}")
    minimum, maximum = float_lines[-2], float_lines[-1]
    return minimum[component], maximum[component]


def main():
    plotfiles = sorted(glob.glob("diags/diag1*"))
    if len(plotfiles) < 2:
        print("FAILED: expected initial and final RZ plotfiles")
        return 1

    bt_min0, bt_max0 = component_minmax(plotfiles[0], 1)
    bt_min1, bt_max1 = component_minmax(plotfiles[-1], 1)
    amplitude0 = max(abs(bt_min0), abs(bt_max0))
    amplitude1 = max(abs(bt_min1), abs(bt_max1))

    print(f"first={plotfiles[0]} last={plotfiles[-1]}")
    print(f"Bt range t=0   = [{bt_min0:.8e}, {bt_max0:.8e}] T")
    print(f"Bt range t=end = [{bt_min1:.8e}, {bt_max1:.8e}] T")

    if not all(math.isfinite(value) for value in (bt_min0, bt_max0, bt_min1, bt_max1)):
        print("FAILED: non-finite Bt value")
        return 1
    if amplitude0 < 1.0e-8:
        print("FAILED: initial Bt was not initialized")
        return 1
    if amplitude1 >= 0.999 * amplitude0:
        print("FAILED: Bt did not damp in the RZ eta=129 smoke")
        return 1

    print("PASSED: RZ hybrid magnetic-diffusion vacuum stability smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())