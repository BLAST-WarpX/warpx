#!/usr/bin/env python3
"""Check RZ magnetic diffusion separately in fluid and EB-covered regions."""

import sys

from analysis_plotfile import all_finite, load_fields, max_abs, plotfiles

R_WALL = 0.015


def main():
    plots = plotfiles()
    if len(plots) < 2:
        print("FAILED: expected initial and final plotfiles")
        return 1

    _, initial = load_fields(plots[0], ("Bt",))
    ds1, final = load_fields(plots[-1], ("Bt",))
    bt0 = initial["Bt"]
    bt1 = final["Bt"]

    if not all_finite(bt0) or not all_finite(bt1):
        print("FAILED: Bt contains NaN/Inf")
        return 1

    def fluid(index):
        return ds1.cell_center(index)[0] < R_WALL

    def covered(index):
        return ds1.cell_center(index)[0] > R_WALL

    initial_amplitude = max_abs(bt0, fluid)
    fluid_amplitude = max_abs(bt1, fluid)
    covered_amplitude = max_abs(bt1, covered)
    print(
        f"Bt amplitudes: initial-fluid={initial_amplitude:.8e}, "
        f"final-fluid={fluid_amplitude:.8e}, final-covered={covered_amplitude:.8e}"
    )

    if initial_amplitude < 1.0e-8:
        print("FAILED: initial Bt was not initialized")
        return 1
    if not 1.0e-8 < fluid_amplitude < 0.999 * initial_amplitude:
        print("FAILED: fluid Bt did not remain finite, nonzero, and damped")
        return 1
    if covered_amplitude > 1.0e-12:
        print("FAILED: EB-covered Bt is not zero")
        return 1
    print("PASSED: fluid Bt damps while EB-covered Bt remains zero")
    return 0


if __name__ == "__main__":
    sys.exit(main())
