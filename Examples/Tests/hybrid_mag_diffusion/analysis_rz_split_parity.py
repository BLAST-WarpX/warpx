#!/usr/bin/env python3
"""RZ Ohm / mag-diff residual-eta parity oracle.

Compares the operator-split run
  (implicit_mag_diffusion=1, mag_diff_eta_explicit_max = eta = 1e-2)
to the pure-explicit reference
  (implicit_mag_diffusion=0, same eta).

With residual eta_D = max(eta - eta_max, 0) the implicit step is a no-op when
eta_max = eta, so final Bt must match the explicit reference.

Without residual subtraction (old double-count), mag-diff still applies full
eta after Ohm already used min(eta, eta_max)=eta, and Bt damps ~twice as much
-> this oracle fails.

Uses Cell_H Bt min/max only (stdlib; no yt/numpy), same style as
analysis_rcyl_parity.py.
"""

import glob
import math
import os
import re
import sys

# Sibling ctest dependency (pure-explicit reference).
REF_DIR = os.path.join("..", "test_rz_hybrid_mag_diffusion_split_parity_ref")
# Loose enough for plotfile float formatting; tight enough that 2x damping fails.
PARITY_TOL = 5.0e-3


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
                values = [
                    float(value)
                    for value in line.strip().rstrip(",").split(",")
                    if value
                ]
            except ValueError:
                continue
            if len(values) >= 3:
                float_lines.append(values)
    if len(float_lines) < 2:
        raise RuntimeError(f"Could not parse field min/max from {cell_h}")
    minimum, maximum = float_lines[-2], float_lines[-1]
    return minimum[component], maximum[component]


def last_plotfile(base_dir):
    plots = sorted(
        p
        for p in glob.glob(os.path.join(base_dir, "diags", "diag1*"))
        if os.path.isdir(p)
    )
    if not plots:
        raise RuntimeError(f"No plotfiles under {base_dir}/diags/")
    return plots[-1]


def first_plotfile(base_dir):
    plots = sorted(
        p
        for p in glob.glob(os.path.join(base_dir, "diags", "diag1*"))
        if os.path.isdir(p)
    )
    if not plots:
        raise RuntimeError(f"No plotfiles under {base_dir}/diags/")
    return plots[0]


def relerr(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1.0e-30)


def main():
    ref_plot0 = first_plotfile(REF_DIR)
    ref_plot = last_plotfile(REF_DIR)
    split_plot0 = first_plotfile(".")
    split_plot = last_plotfile(".")

    # RZ: Br=0, Bt=1, Bz=2
    r0_lo, r0_hi = component_minmax(ref_plot0, 1)
    s0_lo, s0_hi = component_minmax(split_plot0, 1)
    r_lo, r_hi = component_minmax(ref_plot, 1)
    s_lo, s_hi = component_minmax(split_plot, 1)

    amp0_ref = max(abs(r0_lo), abs(r0_hi))
    amp0_split = max(abs(s0_lo), abs(s0_hi))
    amp_ref = max(abs(r_lo), abs(r_hi))
    amp_split = max(abs(s_lo), abs(s_hi))

    amp0_rel = relerr(amp0_ref, amp0_split)
    amp_rel = relerr(amp_ref, amp_split)
    lo_rel = relerr(r_lo, s_lo)
    hi_rel = relerr(r_hi, s_hi)

    print(f"explicit ref t=0  : {ref_plot0}")
    print(f"explicit ref t=end: {ref_plot}")
    print(f"split      t=0  : {split_plot0}")
    print(f"split      t=end: {split_plot}")
    print(
        f"|Bt|_max t=0  : explicit={amp0_ref:.8e}  split={amp0_split:.8e}  "
        f"rel={amp0_rel:.3e}"
    )
    print(
        f"|Bt|_max t=end: explicit={amp_ref:.8e}  split={amp_split:.8e}  "
        f"rel={amp_rel:.3e}"
    )
    print(f"Bt min t=end: explicit={r_lo:.8e}  split={s_lo:.8e}  rel={lo_rel:.3e}")
    print(f"Bt max t=end: explicit={r_hi:.8e}  split={s_hi:.8e}  rel={hi_rel:.3e}")

    values = (amp0_ref, amp0_split, amp_ref, amp_split, r_lo, r_hi, s_lo, s_hi)
    if not all(math.isfinite(v) for v in values):
        print("FAILED: non-finite value")
        return 1
    if amp0_ref < 1.0e-8 or amp0_split < 1.0e-8:
        print("FAILED: initial Bt was not initialized")
        return 1
    # Both must actually damp (resistive physics present on the Ohm path).
    if amp_ref >= 0.999 * amp0_ref:
        print("FAILED: explicit reference did not damp Bt")
        return 1
    if amp_split >= 0.999 * amp0_split:
        print("FAILED: split run did not damp Bt")
        return 1
    if amp0_rel >= PARITY_TOL:
        print(
            f"FAILED: initial |Bt| mismatch {amp0_rel:.3e} >= {PARITY_TOL:.0e} "
            "(setup not comparable)"
        )
        return 1
    if amp_rel >= PARITY_TOL or lo_rel >= PARITY_TOL or hi_rel >= PARITY_TOL:
        print(
            f"FAILED: residual-eta parity (amp {amp_rel:.3e}, min {lo_rel:.3e}, "
            f"max {hi_rel:.3e}) >= {PARITY_TOL:.0e}\n"
            "  Expected: mag-diff with eta_max=eta leaves eta_D=0 and matches "
            "pure-explicit Ohm.\n"
            "  If this fails with large extra damping on the split run, Ohm and "
            "mag-diff are double-counting soft eta."
        )
        return 1

    print(
        f"PASSED: split (eta_max=eta) matches pure-explicit Ohm "
        f"(|Bt| amp + min/max rel < {PARITY_TOL:.0e}); residual-eta partition OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
