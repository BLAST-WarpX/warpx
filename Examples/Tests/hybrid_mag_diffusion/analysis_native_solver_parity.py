#!/usr/bin/env python3
"""Check every magnetic-field cell against the existing GMRES reference."""

import os
import sys

from analysis_plotfile import (
    all_finite,
    last_plotfile,
    load_fields,
    relative_field_norms,
)


def main():
    reference_dir, geometry = sys.argv[1:3]
    names = ("Br", "Bt", "Bz") if geometry == "cylindrical" else ("Bx", "By", "Bz")
    reference_plot = last_plotfile(reference_dir)
    candidate_plot = last_plotfile()
    for plot in [reference_plot, candidate_plot]:
        with open(os.path.join(plot, "Header")) as stream:
            header = stream.read().splitlines()
        count = int(header[1])
        time = float(header[3 + count])
        if plot == reference_plot:
            reference_time = time
        else:
            assert time == reference_time, "Compared different physical times"
    _, reference = load_fields(reference_plot, names)
    _, candidate = load_fields(candidate_plot, names)
    for name in names:
        assert all_finite(reference[name]) and all_finite(candidate[name]), name
        l2, linf = relative_field_norms(reference[name], candidate[name])
        print(f"{name}: relative L2={l2:.3e}, relative Linf={linf:.3e}")
        assert max(l2, linf) < 1.0e-6, f"Full-field solver parity failed for {name}"
    print("PASSED: all magnetic fields match the native GMRES reference")


if __name__ == "__main__":
    main()
