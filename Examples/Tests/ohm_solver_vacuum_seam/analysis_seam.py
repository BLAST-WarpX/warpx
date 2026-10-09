#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
"""Seam-noise comparison for the vacuum-seam switch decision modes.

A magnetized plasma disk drifts through a periodic box; its circular
plasma/vacuum seam samples every angle and cell phase. The legacy per-edge
switch decision lets the E components of a cell take inconsistent
vacuum/plasma decisions along the seam -- the holmstrom vacuum branch when
that treatment is on, the density-floor selection of the guarded Hall term
otherwise -- sourcing a grid-(C4-)patterned spurious E there; the node and
cell decision modes remove most of it.
With one run directory the script only checks boundedness (the guarded
tests use this: their seam ring carries the physical Hall halo of the
floored vacuum, so the modes are pinned by checksums instead of a noise
ratio); with three (holmstrom edge, node, cell) it asserts the reduction.
"""

import os
import re
import sys

import numpy as np

PROB_LO = np.array([-0.8, -0.8])
NCELL = np.array([128, 128])
H = np.array([1.6, 1.6]) / NCELL
STAGGER = {"Ex": (0.5, 0.0), "Ey": (0.0, 0.0), "Ez": (0.0, 0.5)}
R_RING = 0.38  # just outside the disk radius 0.35
NTHETA = 720
STEP = 10


def read_vismf(prefix):
    with open(prefix + "_H") as header_file:
        hdr = header_file.read().split("\n")
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
            ilo, jlo, ihi, jhi, _t0, _t1, ncomp = map(int, m.groups())
            ni, nj = ihi - ilo + 1, jhi - jlo + 1
            data = np.fromfile(f, dtype="<f8", count=ni * nj * ncomp)
            fabs.append(((ilo, jlo), data.reshape(ncomp, nj, ni)[0].T))
    return fabs


def load(plotfile, comp):
    fabs = read_vismf(os.path.join(plotfile, "raw_fields", "Level_0", f"{comp}_fp"))
    shape = tuple(max(lo[d] + a.shape[d] for lo, a in fabs) for d in range(2))
    full = np.full(shape, np.nan)
    for (i0, j0), a in fabs:
        full[i0 : i0 + a.shape[0], j0 : j0 + a.shape[1]] = a
    assert not np.isnan(full).any()
    return full


def ring_metrics(rundir):
    """(max |E| on the seam ring, m4+m8 harmonic amplitude), worst component."""
    plotfile = os.path.join(rundir, "diags", f"diag1{STEP:06d}")
    assert os.path.isdir(plotfile), f"missing {plotfile}"
    th = np.linspace(0, 2 * np.pi, NTHETA, endpoint=False)
    worst_amp, worst_c4 = 0.0, 0.0
    for comp, (sx, sz) in STAGGER.items():
        f = load(plotfile, comp)
        assert np.isfinite(f).all(), f"{rundir}/{comp} not finite"
        fi = (R_RING * np.cos(th) - PROB_LO[0]) / H[0] - sx
        fj = (R_RING * np.sin(th) - PROB_LO[1]) / H[1] - sz
        i0 = np.floor(fi).astype(int)
        j0 = np.floor(fj).astype(int)
        wx, wz = fi - i0, fj - j0
        v = (
            (1 - wx) * (1 - wz) * f[i0, j0]
            + wx * (1 - wz) * f[i0 + 1, j0]
            + (1 - wx) * wz * f[i0, j0 + 1]
            + wx * wz * f[i0 + 1, j0 + 1]
        )
        c = np.fft.rfft(v) / len(v)
        worst_amp = max(worst_amp, float(np.max(np.abs(v))))
        worst_c4 = max(worst_c4, float(2 * np.abs(c[4]) + 2 * np.abs(c[8])))
    return worst_amp, worst_c4


def main():
    runs = sys.argv[1:]
    res = {}
    for r in runs:
        label = os.path.basename(os.path.normpath(r)) or r
        amp, c4 = ring_metrics(r)
        res[label] = (amp, c4)
        print(f"{label}: seam-ring max|E| = {amp:.3e}  m4+m8 = {c4:.3e}")
    if len(runs) == 3:
        edge_amp = list(res.values())[0][0]
        for label, (amp, _c4) in list(res.items())[1:]:
            assert amp <= 0.5 * edge_amp, (
                f"{label}: seam noise {amp:.3e} not below half the edge-mode "
                f"noise {edge_amp:.3e}"
            )
        print("PASS")


if __name__ == "__main__":
    main()
