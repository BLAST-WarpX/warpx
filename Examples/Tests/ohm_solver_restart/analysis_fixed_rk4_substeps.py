#!/usr/bin/env python3

"""
With RKF45 disabled, the B-field sub-stepping uses fixed-step RK4 with exactly
hybrid_pic_model.substeps sub-steps per step. This count must not be adapted,
nor limited by hybrid_pic_model.max_substep_attempts, which only applies to
RKF45. This test sets max_substep_attempts below substeps/2 on purpose.
"""

from pathlib import Path


def substeps(header):
    entries = [
        line.split()
        for line in header.read_text().splitlines()
        if line.startswith("hybrid_pic_substeps ")
    ]
    assert len(entries) == 1 and len(entries[0]) == 2, f"Bad header {header}"
    return int(entries[0][1])


for step in (3, 6):
    count = substeps(Path(f"diags/chk{step:06d}/WarpXHeader"))
    assert count == 40, f"Fixed RK4 substep count changed at step {step}: {count}"
print("Fixed RK4 substep count unchanged: 40")
