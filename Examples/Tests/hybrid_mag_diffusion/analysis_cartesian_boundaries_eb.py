#!/usr/bin/env python3
"""Check Cartesian PEC, pec_insulator feed, and optional stair-step EB support."""

import glob
import math
import os
import re
import sys

B0 = 1.0e-4


def input_dimension():
    with open("warpx_used_inputs") as infile:
        inputs = infile.read()
    match = re.search(r"^amr\.n_cell\s*=\s*([^#\n]+)", inputs, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not read amr.n_cell from warpx_used_inputs")
    return len(match.group(1).split())


def _float_rows(text):
    rows = []
    for line in text.splitlines():
        if not re.match(r"^-?[0-9]", line.strip()):
            continue
        try:
            values = [
                float(value)
                for value in line.strip().rstrip(",").split(",")
                if value.strip()
            ]
        except ValueError:
            continue
        if len(values) >= 3:
            rows.append(values)
    return rows


def component_minmax(plotfile, component):
    cell_h = os.path.join(plotfile, "Level_0", "Cell_H")
    with open(cell_h) as infile:
        text = infile.read()
    sections = re.split(r"^\s*\d+,\s*\d+\s*$", text, flags=re.MULTILINE)
    float_sections = [rows for rows in map(_float_rows, sections) if rows]
    if len(float_sections) < 2:
        raise RuntimeError(f"Could not parse min/max blocks from {cell_h}")
    return (
        min(row[component] for row in float_sections[-2]),
        max(row[component] for row in float_sections[-1]),
    )


def main():
    plotfiles = sorted(glob.glob("diags/diag1*"))
    if len(plotfiles) < 2:
        print("FAILED: expected initial and final plotfiles")
        return 1

    dim = input_dimension()
    feed_component = 0 if dim == 1 else 1  # Bx in 1D_Z; By in XZ/3D
    initial = [component_minmax(plotfiles[0], component) for component in range(3)]
    final = [component_minmax(plotfiles[-1], component) for component in range(3)]
    feed_min, feed_max = final[feed_component]

    print(f"dimension={dim} feed_component={'Bx' if dim == 1 else 'By'}")
    print(f"initial extrema={initial}")
    print(f"final extrema={final}")

    if not all(math.isfinite(value) for extrema in final for value in extrema):
        print("FAILED: magnetic field contains NaN/Inf")
        return 1
    if max(abs(value) for extrema in initial for value in extrema) > 1.0e-12:
        print("FAILED: magnetic field was not initialized to zero")
        return 1
    if not 0.5 * B0 <= feed_max <= 1.5 * B0:
        print("FAILED: pec_insulator did not hold the feed near B0")
        return 1

    if dim == 1:
        if feed_min < 0.1 * B0:
            print(
                "FAILED: feed did not diffuse from the z_hi feed to the z_lo PEC wall"
            )
            return 1
    elif abs(feed_min) > 1.0e-14:
        print("FAILED: interior EB did not keep covered B inert")
        return 1

    print("PASSED: Cartesian hybrid magnetic-diffusion boundary/EB regression")
    return 0


if __name__ == "__main__":
    sys.exit(main())
