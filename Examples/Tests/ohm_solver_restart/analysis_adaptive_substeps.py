#!/usr/bin/env python3

from pathlib import Path

import yt


def substeps(header):
    lines = header.read_text().splitlines()
    entries = [
        line.split() for line in lines if line.startswith("hybrid_pic_substeps ")
    ]
    assert len(entries) == 1, f"Missing or repeated controller state in {header}"
    assert len(entries[0]) == 2, f"Invalid controller state in {header}"
    count = int(entries[0][1])
    assert count >= 2 and count % 2 == 0, f"Invalid substep count in {header}: {count}"
    return count


reference = Path("../test_2d_ohm_solver_adaptive_checkpoint/diags")
saved = substeps(reference / "chk000003/WarpXHeader")
assert saved != 40, "The controller must have adapted before the checkpoint"

# Restart diagnostics are written before any field advance. The input deliberately
# requests 2 substeps, so this also checks that restored state wins over the seed.
restored = substeps(Path("diags/chk000003/WarpXHeader"))
assert restored == saved, f"Controller reset on restart: {saved} -> {restored}"

expected = substeps(reference / "chk000006/WarpXHeader")
continued = substeps(Path("diags/chk000006/WarpXHeader"))
assert continued == expected, (
    f"Controller continuation differs: {expected} != {continued}"
)

# Particle plotfiles use yt's WarpXHeader parser, which accepts only numeric
# species rows at the end of the header. Check both arms, including actual reads.
for plotfile in (reference / "fields000006", Path("diags/fields000006")):
    assert "hybrid_pic_substeps" not in (plotfile / "WarpXHeader").read_text()
    ds = yt.load(str(plotfile))
    assert ("ions", "particle_weight") in ds.field_list
    assert ds.all_data()["ions", "particle_weight"].size == 16 * 16 * 2 * 2

print(f"Adaptive substeps restored exactly: {saved}; continued to {continued}")
