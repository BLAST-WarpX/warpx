#!/usr/bin/env python3
"""Check native/Pmat identity rows and physical damping at PEC endwalls."""

import json
import math
import sys
from pathlib import Path

from analysis_rz_pec_feed import main as check_feed
from analysis_rz_z_pec_wall import main as check_fields

if __name__ == "__main__":
    audit = json.loads(Path("pmat_audit.json").read_text())
    evolved = len(sys.argv) > 1 and sys.argv[1] == "evolved"
    if evolved:
        assert audit["clamped_rows"] == 0, (
            "An evolved insulator was incorrectly clamped"
        )
    else:
        assert audit["clamped_rows"] > 0, "No native PEC boundary rows were exercised"
    for name in ["boundary_P_error", "boundary_A_error"]:
        assert math.isfinite(audit[name]) and audit[name] <= 1.0e-12, audit
    assert math.isfinite(audit["relative_Pmat_vs_shell"]), audit
    assert audit["relative_Pmat_vs_shell"] < 1.0e-12, audit
    print("PASSED: assembled and native operators use the declared boundary map")
    feed = len(sys.argv) > 1 and sys.argv[1] == "feed"
    raise SystemExit(check_feed() if feed else check_fields())
