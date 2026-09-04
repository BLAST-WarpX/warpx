#!/usr/bin/env python3
"""Compare loaded external fields against the analytic equilibrium.

For each plotfile (written with diag1.plot_raw_fields = 1) the raw Yee
components are compared with the analytic screw-pinch fields evaluated
at each component's own staggered positions. The pointwise error field
is then sampled on rings and decomposed azimuthally.

Because the equilibrium is axisymmetric and the reader's resampling
stencil is invariant under 90-degree rotations (which swap x and y and
the corresponding field components), the error of the reconstructed
cylindrical fields E_r, B_theta and B_z contains only m = 0, 4, 8, ...
harmonics: m=0 is the isotropic smoothing bias, m=4 is the anisotropic
(C4v) imprint of the per-component staggering stencil.

Usage:
  analysis_m4.py label=plotfile [label=plotfile ...]
      print the decomposition for each case, plus cross-case m=4 ratios

  analysis_m4.py --case {yee,nodal} plotfile
      CI mode: analyze one case and assert its expected properties
      (measured values at 64x64x8 with h = a/10, with wide margins):
      - yee: exact sampling, every component matches the analytic
        equilibrium to near round-off
      - nodal: the n-linear resampling shows the m=4 anisotropy
"""

import sys

import numpy as np
import yt
from equilibrium import (
    FIELD_FUNCS,
    PROB_LO,
    YEE_POSITION,
    A,
    H,
    btheta_over_r,
    bz_of_u,
    er_over_r,
)

yt.set_log_level(50)

RADII = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
NTHETA = 720
COMPS = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]


def load_raw_errors(plotfile):
    """Return {comp: (err2d, num2d, frac_pos)} on the z-averaged x-y lattice."""
    ds = yt.load(plotfile)

    out = {}
    for c in COMPS:
        # assemble the (possibly box-decomposed) raw field on the full lattice
        pieces = []
        for g in ds.index.grids:
            raw = g["raw", f"{c}_fp"].v
            assert raw.ndim == 4, f"unexpected raw shape {raw.shape} for {c}"
            i0 = np.rint((g.LeftEdge.v - PROB_LO) / H).astype(int)
            pieces.append((i0, raw[..., 0]))
        shape = tuple(max(i0[d] + r.shape[d] for i0, r in pieces) for d in range(3))
        num3d = np.full(shape, np.nan)
        for i0, r in pieces:
            num3d[
                i0[0] : i0[0] + r.shape[0],
                i0[1] : i0[1] + r.shape[1],
                i0[2] : i0[2] + r.shape[2],
            ] = r
        assert not np.isnan(num3d).any()
        frac = np.array(YEE_POSITION[c])
        nx, ny, nz = num3d.shape
        x = PROB_LO[0] + (np.arange(nx) + frac[0]) * H[0]
        y = PROB_LO[1] + (np.arange(ny) + frac[1]) * H[1]
        z = PROB_LO[2] + (np.arange(nz) + frac[2]) * H[2]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        ref3d = FIELD_FUNCS[c](X, Y, Z)
        err3d = num3d - ref3d
        # equilibrium is z-invariant: average over z
        out[c] = (err3d.mean(axis=2), num3d.mean(axis=2), frac)
    return out


def ring_sample(F2d, frac, r0, ntheta=NTHETA):
    """Bilinear sample of a 2D lattice field on a ring of radius r0."""
    th = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    fi = (r0 * np.cos(th) - PROB_LO[0]) / H[0] - frac[0]
    fj = (r0 * np.sin(th) - PROB_LO[1]) / H[1] - frac[1]
    i0 = np.floor(fi).astype(int)
    j0 = np.floor(fj).astype(int)
    wx = fi - i0
    wy = fj - j0
    v = (
        (1 - wx) * (1 - wy) * F2d[i0, j0]
        + wx * (1 - wy) * F2d[i0 + 1, j0]
        + (1 - wx) * wy * F2d[i0, j0 + 1]
        + wx * wy * F2d[i0 + 1, j0 + 1]
    )
    return th, v


def harmonics(v):
    """Return (mean, {m: amplitude}) of a periodic signal."""
    c = np.fft.rfft(v) / len(v)
    amps = {m: 2.0 * np.abs(c[m]) for m in range(1, 9)}
    return np.abs(c[0].real), amps


def analyze(plotfile, label):
    data = load_raw_errors(plotfile)

    print(f"\n=== case '{label}'  ({plotfile}) ===")
    print("  pointwise |error| (max over grid, normalized per field):")
    escale = max(np.abs(data[c][1]).max() for c in ("Ex", "Ey"))
    bscale = max(np.abs(data[c][1]).max() for c in ("Bx", "By"))
    dbz = abs(bz_of_u(np.array(1.0e9)) - bz_of_u(np.array(0.0)))
    scales = {
        "Ex": escale,
        "Ey": escale,
        "Ez": escale,
        "Bx": bscale,
        "By": bscale,
        "Bz": dbz,
    }
    for c in COMPS:
        err = np.abs(data[c][0]).max()
        print(f"    {c}: {err:11.4e}   rel {err / scales[c]:11.4e}")

    print("  ring decomposition of the error (normalized amplitudes):")
    print(
        f"   {'r':>5} | {'Er m0':>9} {'Er m4':>9} | {'Bth m0':>9} {'Bth m4':>9} | {'Bz m0':>9} {'Bz m4':>9}"
    )
    rows = {}
    for r0 in RADII:
        u0 = (r0 / A) ** 2
        th, ex = ring_sample(data["Ex"][0], data["Ex"][2], r0)
        _, ey = ring_sample(data["Ey"][0], data["Ey"][2], r0)
        _, bx = ring_sample(data["Bx"][0], data["Bx"][2], r0)
        _, by = ring_sample(data["By"][0], data["By"][2], r0)
        _, bz = ring_sample(data["Bz"][0], data["Bz"][2], r0)

        er = ex * np.cos(th) + ey * np.sin(th)
        bth = -bx * np.sin(th) + by * np.cos(th)

        er_ref = abs(r0 * er_over_r(np.array(u0)))
        bth_ref = abs(r0 * btheta_over_r(np.array(u0)))

        row = {}
        for name, sig, ref in (
            ("Er", er, er_ref),
            ("Bth", bth, bth_ref),
            ("Bz", bz, dbz),
        ):
            m0, amps = harmonics(sig)
            row[f"{name}_m0"] = m0 / ref
            row[f"{name}_m4"] = amps[4] / ref
        rows[r0] = row
        print(
            f"   {r0:5.2f} | {row['Er_m0']:9.2e} {row['Er_m4']:9.2e}"
            f" | {row['Bth_m0']:9.2e} {row['Bth_m4']:9.2e}"
            f" | {row['Bz_m0']:9.2e} {row['Bz_m4']:9.2e}"
        )
    return rows


def assert_case(case, plotfile):
    data = load_raw_errors(plotfile)
    rows = analyze(plotfile, case)

    if case == "yee":
        # position attribute honored: identity sampling up to round-off
        escale = max(np.abs(data[c][1]).max() for c in ("Ex", "Ey"))
        bscale = max(np.abs(data[c][1]).max() for c in ("Bx", "By", "Bz"))
        for c in ("Ex", "Ey", "Ez"):
            assert np.abs(data[c][0]).max() <= 1.0e-12 * escale, c
        for c in ("Bx", "By", "Bz"):
            assert np.abs(data[c][0]).max() <= 1.0e-12 * bscale, c
    elif case == "nodal":
        # the n-linear half-cell resampling imprints an m=4 mode
        row = rows[0.30]
        assert 5.0e-5 <= row["Er_m4"] <= 1.0e-3, row["Er_m4"]
        assert 1.5e-4 <= row["Bth_m4"] <= 2.0e-3, row["Bth_m4"]
        assert row["Er_m0"] <= 1.0e-2, row["Er_m0"]
    else:
        raise ValueError(f"unknown case '{case}'")
    print(f"PASS: case '{case}'")


if __name__ == "__main__":
    if sys.argv[1] == "--case":
        assert_case(sys.argv[2], sys.argv[3])
        sys.exit(0)

    cases = {}
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        cases[label] = analyze(path, label)

    labels = list(cases)
    if len(labels) > 1:
        base = labels[0]
        for other in labels[1:]:
            print(f"\n=== m=4 ratio: {other} / {base} ===")
            print(f"   {'r':>5} | {'Er m4':>9} | {'Bth m4':>9} | {'Bz m4':>9}")
            for r0 in RADII:
                a, b = cases[base][r0], cases[other][r0]

                def ratio(k):
                    return b[k] / a[k] if a[k] > 0 else float("nan")

                print(
                    f"   {r0:5.2f} | {ratio('Er_m4'):9.3f}"
                    f" | {ratio('Bth_m4'):9.3f} | {ratio('Bz_m4'):9.3f}"
                )
