#!/usr/bin/env python3
"""Check Cartesian PEC/feed boundaries and spatially identify covered EB cells."""

import re
import sys

from analysis_plotfile import all_finite, load_fields, max_abs, plotfiles

B0 = 1.0e-4


def input_dimension():
    with open("warpx_used_inputs") as infile:
        inputs = infile.read()
    match = re.search(r"^amr\.n_cell\s*=\s*([^#\n]+)", inputs, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not read amr.n_cell")
    return len(match.group(1).split())


def main():
    plots = plotfiles()
    if len(plots) < 2:
        print("FAILED: expected initial and final plotfiles")
        return 1

    dim = input_dimension()
    names = ("Bx", "By", "Bz")
    _, initial = load_fields(plots[0], names)
    ds, final = load_fields(plots[-1], names)
    if any(not all_finite(field) for field in final.values()):
        print("FAILED: magnetic field contains NaN/Inf")
        return 1
    if max(max_abs(field) for field in initial.values()) > 1.0e-12:
        print("FAILED: magnetic field was not initialized to zero")
        return 1

    feed_name = "Bx" if dim == 1 else "By"
    feed = final[feed_name]
    feed_face = max_abs(feed, lambda index: index[0] == ds.domain_dimensions[0] - 1)
    if not 0.5 * B0 <= feed_face <= 1.5 * B0:
        print(
            f"FAILED: pec_insulator feed slice is {feed_face:.8e}, expected near {B0:.8e}"
        )
        return 1

    if dim > 1:

        def covered(index):
            x, second, third = ds.cell_center(index)
            level_set = 0.04 - (x - 0.5) ** 2 - (second - 0.5) ** 2
            if dim == 3:
                level_set -= (third - 0.5) ** 2
            return level_set > 0.0

        covered_amplitude = max(max_abs(field, covered) for field in final.values())
        print(f"feed-slice={feed_face:.8e}, covered |B|max={covered_amplitude:.8e}")
        if covered_amplitude > 1.0e-12:
            print("FAILED: EB-covered magnetic field is not zero")
            return 1
    else:
        print(f"feed-slice={feed_face:.8e}")

    print("PASSED: Cartesian feed and spatial EB checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
