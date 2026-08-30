#!/usr/bin/env python3
"""RZ hybrid magnetic diffusion with a Dirichlet B_t feed (pec_insulator),
Crank-Nicolson (theta=0.5), stiff vacuum eta (129 Ohm m).

CN is unconditionally stable but NOT L-stable: with dt >> R^2/chi (stiff), the
high-diffusion modes have amplification g -> -1, so they reach the diffusion
equilibrium only slowly (they oscillate rather than being killed in one step,
as the L-stable backward-Euler scheme would). The companion theta=1 test
(test_rz_hybrid_mag_diffusion_pec_feed) therefore holds the feed face near B0
within a few steps; this theta=0.5 smoke does NOT, and that is expected.

This smoke checks the feed is correctly imposed by the affine split -- not
steady state:
  - B_t grew from the zero initialization (the feed injected field)
  - near-axis B_t stays well below the feed value (axis regularity)
  - the run is finite/stable (no NaN/abort)

It deliberately does NOT assert the feed face is held at B0 (a steady-state
check appropriate for L-stable backward Euler, not for stiff CN). See the
design note (notes/2026-07-17_theta_method_variable_eta_design.md, Sec. 7).
"""

import glob
import math
import os
import re
import sys

B0 = 1.0e-4  # saturated feed value [T] (insulator.By_x_hi for t >= 1 ns)


def _float_rows(text):
    rows = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if re.match(r"^-?[0-9]", line):
            try:
                values = [float(v) for v in line.split(",") if v.strip()]
            except ValueError:
                continue
            if len(values) >= 3:
                rows.append(values)
    return rows


def component_minmax(diag_dir, component):
    """Global (min, max) of a component across all fabs (mirrors
    analysis_rz_pec_feed.py: min over the per-fab MIN block, max over the MAX
    block)."""
    cell_h = os.path.join(diag_dir, "Level_0", "Cell_H")
    if not os.path.isfile(cell_h):
        raise FileNotFoundError(cell_h)
    with open(cell_h) as fh:
        text = fh.read()
    sections = re.split(r"^\s*\d+,\s*\d+\s*$", text, flags=re.MULTILINE)
    float_sections = [s for s in (_float_rows(sec) for sec in sections) if s]
    if len(float_sections) < 2:
        raise RuntimeError(f"Could not parse min/max blocks from {cell_h}")
    min_rows = float_sections[-2]
    max_rows = float_sections[-1]
    gmin = min(r[component] for r in min_rows)
    gmax = max(r[component] for r in max_rows)
    return gmin, gmax


def main():
    plotfiles = sorted(glob.glob("diags/diag1*"))
    if len(plotfiles) < 2:
        print("FAILED: expected initial and final RZ plotfiles")
        return 1

    # component 1 = Bt (toroidal) in RZ plot output (Br, Bt, Bz).
    bt_min0, bt_max0 = component_minmax(plotfiles[0], 1)
    bt_min1, bt_max1 = component_minmax(plotfiles[-1], 1)

    print(f"first={plotfiles[0]} last={plotfiles[-1]}")
    print(f"Bt t=0   min={bt_min0:.8e}  max={bt_max0:.8e} T")
    print(f"Bt t=end min={bt_min1:.8e}  max={bt_max1:.8e} T")

    if not all(math.isfinite(v) for v in (bt_min0, bt_max0, bt_min1, bt_max1)):
        print("FAILED: non-finite Bt value")
        return 1

    amp0 = max(abs(bt_min0), abs(bt_max0))
    amp1 = max(abs(bt_min1), abs(bt_max1))

    # The feed must inject field: amplitude grew from the zero initialization.
    # (CN reaches the equilibrium slowly when stiff, so we only require growth,
    # not that it has reached B0.)
    if amp1 <= 1.0001 * max(amp0, 1.0e-30):
        print("FAILED: Bt amplitude did not grow under the feed")
        return 1
    # Axis regularity: |Bt| on the axis (the min side) stays well below the
    # feed value. (Looser than the theta=1 test -- stiff CN has not settled, so
    # we only require the axis not blow up toward the feed value.)
    if abs(bt_min1) > 0.2 * B0:
        print(f"FAILED: axis |Bt| too large (min_end={bt_min1:.3e})")
        return 1
    # The feed face should already carry a measurable fraction of B0 (CN injects
    # the feed even if it equilibrates slowly); require it be non-negligible.
    if amp1 < 1.0e-3 * B0:
        print(f"FAILED: feed face Bt too small (max_end={bt_max1:.3e})")
        return 1

    print(
        "PASSED: RZ hybrid magnetic-diffusion Dirichlet B_t feed, "
        "Crank-Nicolson stiff smoke (finite, feed-injected, axis-regular)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
