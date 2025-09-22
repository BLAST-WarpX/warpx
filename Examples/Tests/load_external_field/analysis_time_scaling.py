#!/usr/bin/env python3
# Compare B(component) scaling between two plotfiles:
# checks that BN/B0 ~= expected_ratio on cells where |B0| > min_abs

import argparse

import numpy as np


def load_plotfile_arrays(path, comp):
    import yt

    ds = yt.load(path)
    if hasattr(ds, "force_periodicity"):
        ds.force_periodicity()
    ad = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    arr = ad[("boxlib", comp)].squeeze().v
    t = float(ds.current_time.in_units("s"))
    return arr, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pf0", required=True, help="plotfile at t0 (e.g. diags/diag1000000)"
    )
    p.add_argument(
        "--pfN", required=True, help="plotfile at tN (e.g. diags/diag1000300)"
    )
    p.add_argument("--component", default="Bz", help="Bx|By|Bz (default: Bz)")
    p.add_argument(
        "--expected_ratio",
        type=float,
        required=True,
        help="Expected BN/B0 ratio (e.g. 0.5)",
    )
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=0.0)
    p.add_argument(
        "--min_abs",
        type=float,
        default=0.0,
        help="Ignore cells with |B0| <= min_abs to avoid 0/0",
    )
    args = p.parse_args()

    B0, t0 = load_plotfile_arrays(args.pf0, args.component)
    BN, tN = load_plotfile_arrays(args.pfN, args.component)

    if B0.shape != BN.shape:
        raise RuntimeError(f"Shape mismatch: {B0.shape} vs {BN.shape}")

    mask = np.abs(B0) > args.min_abs
    if not np.any(mask):
        raise RuntimeError("Mask is empty; increase --min_abs or check data.")

    ratio = np.zeros_like(B0, dtype=float)
    ratio[mask] = BN[mask] / B0[mask]
    r_mean = float(np.mean(ratio[mask]))
    r_med = float(np.median(ratio[mask]))

    print(f"t0 = {t0:.9e} s, tN = {tN:.9e} s")
    print(f"Expected ratio = {args.expected_ratio:.8f}")
    print(f"Observed ratio: mean = {r_mean:.8f}, median = {r_med:.8f}")

    assert np.isclose(r_med, args.expected_ratio, rtol=args.rtol, atol=args.atol), (
        f"Median ratio {r_med} != expected {args.expected_ratio} (rtol={args.rtol}, atol={args.atol})"
    )
    # softer check
    assert np.isclose(
        r_mean, args.expected_ratio, rtol=10 * args.rtol, atol=args.atol
    ), f"Mean ratio {r_mean} != expected {args.expected_ratio}"


if __name__ == "__main__":
    main()
