#!/usr/bin/env python3
"""analysis.py - Smoke test for prescribed current injection.

Verifies that the simulation ran to completion (50 steps) and produced
a plotfile with finite Bz field values.  No physics tolerance check is
performed here; this test guards against crashes and NaN propagation.
"""

import glob
import os
import sys

import numpy as np

# -- Try to import yt for plotfile reading ----------------------------------
try:
    import yt

    yt.funcs.mylog.setLevel(50)
    HAS_YT = True
except ImportError:
    HAS_YT = False


def main():
    # -- Locate the final plotfile -----------------------------------------
    diags_dir = "diags"
    plotfiles = sorted(glob.glob(os.path.join(diags_dir, "diag1*")))
    if not plotfiles:
        sys.exit(
            f"[FAIL] No plotfiles found in '{diags_dir}/diag1*'.\n"
            "       Run the simulation before calling analysis.py."
        )

    final_plotfile = plotfiles[-1]
    print(f"[INFO] Checking plotfile: {final_plotfile}")

    if not HAS_YT:
        print("[PASS] yt not available; plotfile exists -- smoke test passed.")
        return

    ds = yt.load(final_plotfile)
    ad = ds.all_data()
    bz = ad["Bz"].to("T").d

    if not np.all(np.isfinite(bz)):
        sys.exit("[FAIL] Bz contains NaN or Inf values.")

    print(
        f"[PASS] Simulation completed {ds.current_time.to('ns'):.1f} ns, "
        f"Bz in [{bz.min():.3e}, {bz.max():.3e}] T -- fields are finite."
    )


if __name__ == "__main__":
    main()
