#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Verify full-domain (including boundary-shell) div(B) and PEC/PMC BCs."""

import sys

import numpy as np
import yt

B_REF = 1704.509018
H_CELL = 1.5e-6


def check_plotfile(path: str, bc_mode: str) -> None:
    ds = yt.load(path)
    ad = ds.covering_grid(
        level=0,
        left_edge=ds.domain_left_edge,
        dims=ds.domain_dimensions,
    )
    divb = ad["boxlib", "divB"].v
    bx = ad["boxlib", "Bx"].v
    by = ad["boxlib", "By"].v
    bz = ad["boxlib", "Bz"].v

    max_divb_norm = float(np.max(np.abs(divb))) * H_CELL / B_REF
    max_b_norm = (
        float(max(np.max(np.abs(bx)), np.max(np.abs(by)), np.max(np.abs(bz)))) / B_REF
    )

    print(
        f"{path} ({bc_mode}): h*max|divB|/B_ref = {max_divb_norm:.3e}, "
        f"max|B|/B_ref = {max_b_norm:.3e}"
    )

    assert max_b_norm > 0.1, (
        f"Cleaned magnetic field unexpectedly vanished ({max_b_norm})"
    )
    assert max_divb_norm < 1.0e-10, (
        f"Full-domain div(B) (including boundary shell) too large: {max_divb_norm:.3e}"
    )

    if bc_mode == "pmc":
        # Check tangential cell-centered B near walls (Dirichlet phi preserves
        # small near-wall |B_t|; inverted Neumann phi amplifies it > 0.5 B_ref).
        bt_wall = (
            max(
                float(np.max(np.abs(bx[:, [0, -1], :]))),
                float(np.max(np.abs(bx[:, :, [0, -1]]))),
                float(np.max(np.abs(by[[0, -1], :, :]))),
                float(np.max(np.abs(by[:, :, [0, -1]]))),
                float(np.max(np.abs(bz[[0, -1], :, :]))),
                float(np.max(np.abs(bz[:, [0, -1], :]))),
            )
            / B_REF
        )
        print(f"{path} (pmc): near-wall max|B_t|/B_ref = {bt_wall:.3e}")
        assert bt_wall < 0.15, (
            f"PMC near-wall tangential |B_t|/B_ref too large: {bt_wall:.3e}"
        )


def main() -> None:
    bc_mode = sys.argv[1]
    if bc_mode not in ("pec", "pmc"):
        raise ValueError(f"Expected 'pec' or 'pmc', got {bc_mode!r}")

    check_plotfile("diags/diag1000000", bc_mode)
    check_plotfile("diags/diag1000004", bc_mode)


if __name__ == "__main__":
    main()
