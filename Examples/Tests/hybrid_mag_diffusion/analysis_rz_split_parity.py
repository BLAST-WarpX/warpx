#!/usr/bin/env python3
"""Full-field comparison of explicit and explicit/implicit resistivity splits."""

import os
import re
import sys

from analysis_plotfile import (
    all_finite,
    load_fields,
    max_abs,
    plotfiles,
    relative_field_norms,
)

DEFAULT_REF_DIR = os.path.join("..", "test_rz_hybrid_mag_diffusion_split_parity_ref")
PARITY_TOL = 5.0e-3


def field_vector(fields, names):
    return {
        (name, index): value for name in names for index, value in fields[name].items()
    }


def cartesian_run():
    with open("warpx_used_inputs") as infile:
        return not re.search(r"^geometry\.dims\s*=\s*R", infile.read(), re.MULTILINE)


def require_partial_split():
    with open("warpx_used_inputs") as infile:
        inputs = infile.read()

    def scalar(key):
        match = re.search(rf"^{re.escape(key)}\s*=\s*([^#\s]+)", inputs, re.MULTILINE)
        if not match:
            raise RuntimeError(f"Could not read {key}")
        return float(match.group(1))

    eta = scalar("hybrid_pic_model.mag_diff_constant_eta")
    eta_explicit = scalar("hybrid_pic_model.mag_diff_eta_explicit_max")
    if not 0.0 < eta_explicit < eta:
        raise RuntimeError(
            f"Partial split requires 0 < eta_explicit_max < eta, got {eta_explicit}, {eta}"
        )


def main():
    reference_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REF_DIR
    if len(sys.argv) > 2 and sys.argv[2] == "require_partial":
        require_partial_split()
    reference_plots = plotfiles(reference_dir)
    split_plots = plotfiles()
    if len(reference_plots) < 2 or len(split_plots) < 2:
        print("FAILED: expected initial and final plotfiles for both runs")
        return 1

    names = ("Bx", "By", "Bz") if cartesian_run() else ("Br", "Bt", "Bz")
    _, ref_initial = load_fields(reference_plots[0], names)
    _, split_initial = load_fields(split_plots[0], names)
    _, ref_final = load_fields(reference_plots[-1], names)
    _, split_final = load_fields(split_plots[-1], names)
    arrays = (
        *ref_initial.values(),
        *split_initial.values(),
        *ref_final.values(),
        *split_final.values(),
    )
    if any(not all_finite(array) for array in arrays):
        print("FAILED: magnetic field contains NaN/Inf")
        return 1

    initial_l2, initial_linf = relative_field_norms(
        field_vector(ref_initial, names), field_vector(split_initial, names)
    )
    final_l2, final_linf = relative_field_norms(
        field_vector(ref_final, names), field_vector(split_final, names)
    )
    transverse_name = "By" if cartesian_run() else "Bt"
    initial_amplitude = max_abs(ref_initial[transverse_name])
    final_amplitude = max_abs(ref_final[transverse_name])
    print(
        f"initial: relative L2={initial_l2:.3e}, Linf={initial_linf:.3e}; "
        f"final: relative L2={final_l2:.3e}, Linf={final_linf:.3e}"
    )
    print(
        f"reference |{transverse_name}|max: "
        f"{initial_amplitude:.8e} -> {final_amplitude:.8e}"
    )

    if initial_amplitude < 1.0e-8 or final_amplitude >= 0.999 * initial_amplitude:
        print("FAILED: reference Bt was not initialized and damped")
        return 1
    if max(initial_l2, initial_linf, final_l2, final_linf) >= PARITY_TOL:
        print(f"FAILED: full-field split parity error >= {PARITY_TOL:.0e}")
        return 1
    print(f"PASSED: full-field resistivity-split parity < {PARITY_TOL:.0e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
