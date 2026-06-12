#!/usr/bin/env python3
#
# --- Cross-box seam equivalence of the conformal (ECT) embedded-boundary
# --- update: the same resistive-diffusion problem, decomposed along z so
# --- that box seams cross the conformal borrowing planes, must reproduce
# --- the y-split run (whose seams never cross the borrowing planes) to
# --- communication-roundoff accuracy. Before the cross-box reduction of
# --- the face-extension passes, seam faces silently lost borrowed area and
# --- Venl contributions and this comparison failed at the percent level.

import argparse

import numpy as np
import yt

yt.set_log_level(50)

TOL_REL = 1.0e-10
FIELDS = ("Bx", "By", "Bz", "jx", "jy", "jz")


def load(path):
    ds = yt.load(path)
    ad = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return {name: np.asarray(ad[("boxlib", name)]) for name in FIELDS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="z-split (seam) plotfile")
    parser.add_argument("--ref", required=True, help="y-split reference plotfile")
    args = parser.parse_args()

    seam = load(args.path)
    ref = load(args.ref)

    failures = []
    for name in FIELDS:
        scale = max(float(np.max(np.abs(ref[name]))), 1e-300)
        rel = float(np.max(np.abs(seam[name] - ref[name]))) / scale
        status = "PASS" if rel <= TOL_REL else "FAIL"
        print(f"[{status}] seam equivalence {name}: max rel diff = {rel:.3e}")
        if rel > TOL_REL:
            failures.append(name)

    assert not failures, (
        f"cross-box seam run deviates from the seam-free reference in "
        f"{failures} beyond {TOL_REL} relative"
    )
    print("\nall seam-equivalence checks passed")


if __name__ == "__main__":
    main()
