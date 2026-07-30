#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
"""Consistency checks for the hybrid conformal (ECT) embedded-boundary wall.

The deck embeds a rotated-square PEC block in a uniform-B, drifting-ion
hybrid run. The constitutive closure zeroes the Ohm's-law E and the current
on every covered and cut edge, so deep inside the conductor both must be
exactly zero, while the fluid keeps finite fields and its uniform B. Raw
(staggered) fields are read directly from the plotfile VisMF sets.
"""

import os
import re
import sys

import numpy as np

# grid constants mirroring inputs_test_2d_hybrid_conformal_wall
THETA = np.pi / 8
AHALF = 0.25  # conductor: rotated square, max(|xr|,|zr|) < AHALF
B0 = 0.02
B1 = 0.005  # external coil By at the final step (ramp reaches 1 at t = 1.5 ns)
PROB_LO = np.array([-0.8, -0.8])
NCELL = np.array([32, 32])
H = 1.6 / NCELL

STAGGER = {
    "Ex": (0.5, 0.0),
    "Ey": (0.0, 0.0),
    "Ez": (0.0, 0.5),
    "Bx": (0.0, 0.5),
    "By": (0.5, 0.5),
    "Bz": (0.5, 0.0),
    "jx": (0.5, 0.0),
    "jy": (0.0, 0.0),
    "jz": (0.0, 0.5),
}


def read_vismf(prefix):
    """All FABs of a VisMF set -> list of ((ilo, jlo), data2d)."""
    hdr = open(prefix + "_H").read().split("\n")
    nfabs = int(hdr[4].split()[0].lstrip("("))
    fod = [line for line in hdr if line.startswith("FabOnDisk:")]
    fabs = []
    for line in fod[:nfabs]:
        _, fname, off = line.split()
        with open(os.path.join(os.path.dirname(prefix), fname), "rb") as f:
            f.seek(int(off))
            header = b""
            while not header.endswith(b"\n"):
                header += f.read(1)
            m = re.search(
                rb"\(\((-?\d+),(-?\d+)\) \((-?\d+),(-?\d+)\) \((\d+),(\d+)\)\)\s+(\d+)",
                header,
            )
            ilo, jlo, ihi, jhi, _ti, _tj, ncomp = map(int, m.groups())
            ni, nj = ihi - ilo + 1, jhi - jlo + 1
            data = np.fromfile(f, dtype="<f8", count=ni * nj * ncomp)
            fabs.append(((ilo, jlo), data.reshape(ncomp, nj, ni)[0].T))
    return fabs


def load_raw(plotfile, comp):
    """Assemble the raw staggered field and its point coordinates."""
    fabs = read_vismf(os.path.join(plotfile, "raw_fields", "Level_0", f"{comp}_fp"))
    shape = tuple(max(lo[d] + a.shape[d] for lo, a in fabs) for d in range(2))
    full = np.full(shape, np.nan)
    for (i0, j0), a in fabs:
        full[i0 : i0 + a.shape[0], j0 : j0 + a.shape[1]] = a
    assert not np.isnan(full).any(), f"{comp}: holes in raw assembly"
    sx, sz = STAGGER[comp]
    x = PROB_LO[0] + (np.arange(shape[0]) + sx) * H[0]
    z = PROB_LO[1] + (np.arange(shape[1]) + sz) * H[1]
    X, Z = np.meshgrid(x, z, indexing="ij")
    return full, X, Z


def main():
    plotfile = sys.argv[1]
    hmax = float(np.max(H))
    ok = True
    for comp in STAGGER:
        f, X, Z = load_raw(plotfile, comp)
        assert np.isfinite(f).all(), f"{comp} not finite"
        xr = X * np.cos(THETA) + Z * np.sin(THETA)
        zr = -X * np.sin(THETA) + Z * np.cos(THETA)
        deep = np.maximum(np.abs(xr), np.abs(zr)) < AHALF - 1.5 * hmax
        if comp[0] in ("E", "j") and deep.any():
            worst = float(np.max(np.abs(f[deep])))
            print(f"{comp}: max deep-conductor |{comp}| = {worst:.3e}")
            if worst != 0.0:
                ok = False
    by, X, Z = load_raw(plotfile, "By")
    xr = X * np.cos(THETA) + Z * np.sin(THETA)
    zr = -X * np.sin(THETA) + Z * np.cos(THETA)
    box = np.maximum(np.abs(xr), np.abs(zr))
    fluid = box > AHALF + 1.5 * hmax
    deep = box < AHALF - 1.5 * hmax
    # the quasi-static coil field reaches the fluid, while covered faces
    # exclude the external flux entirely: deep-conductor By stays exactly B0
    prof = np.cos(2 * np.pi * Z / 1.6)
    assert np.all(np.abs(by[fluid] - (B0 + B1 * prof[fluid])) < 0.2 * B1), (
        "fluid coil field wrong"
    )
    assert np.all(np.abs(by[deep] - B0) < 1.0e-14 * B0), (
        "external flux leaked into covered faces"
    )
    ex, _, _ = load_raw(plotfile, "Ex")
    assert np.max(np.abs(ex)) > 0.0, "Ohm E identically zero (wall inactive?)"
    assert ok, "constitutive PEC violated: nonzero E or j deep in the conductor"
    print("PASS")


if __name__ == "__main__":
    main()
