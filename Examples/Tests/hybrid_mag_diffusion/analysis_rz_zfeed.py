#!/usr/bin/env python3
"""Check that an axial pec_insulator Bt feed fills the RZ domain."""

import sys

from analysis_plotfile import all_finite, load_fields, max_abs, plotfiles

B0 = 1.0e-4


def main():
    plots = plotfiles()
    if len(plots) < 2:
        print("FAILED: expected initial and final plotfiles")
        return 1

    _, initial = load_fields(plots[0], ("Bt",))
    _, final = load_fields(plots[-1], ("Bt",))
    bt0 = initial["Bt"]
    bt1 = final["Bt"]
    if not all_finite(bt0) or not all_finite(bt1):
        print("FAILED: Bt contains NaN/Inf")
        return 1
    if max_abs(bt0) > 1.0e-12:
        print("FAILED: expected near-zero initial Bt")
        return 1

    peak = max(bt1.values())
    if not 0.5 * B0 <= peak <= 1.5 * B0:
        print(f"FAILED: final peak Bt={peak:.8e} is not near the feed scale")
        return 1

    nz = max(index[1] for index in bt1) + 1
    bt_z = [max_abs(bt1, lambda index, j=j: index[1] == j) for j in range(nz)]
    feed = bt_z[0]
    middle = bt_z[len(bt_z) // 2]
    ratio = middle / max(feed, 1.0e-30)
    print(
        f"peak={peak:.8e}, z-feed={feed:.8e}, z-middle={middle:.8e}, ratio={ratio:.4f}"
    )
    if ratio < 0.5:
        print("FAILED: Bt did not fill the axial interior")
        return 1
    print("PASSED: axial Bt feed reaches the domain interior")
    return 0


if __name__ == "__main__":
    sys.exit(main())
