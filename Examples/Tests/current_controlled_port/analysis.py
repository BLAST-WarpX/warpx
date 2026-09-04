#!/usr/bin/env python3
"""Verify that the 3D port projection preserves the magnetic constraint."""

from pathlib import Path

import numpy as np
import yt

yt.funcs.mylog.setLevel(50)

plotfiles = sorted(
    path
    for path in Path("diags").glob("diag1*")
    if path.name.removeprefix("diag1").isdigit()
)
assert plotfiles
ds = yt.load(str(plotfiles[-1]))
data = ds.all_data()


def field(name):
    matches = [item for item in ds.field_list if item[1] == name]
    assert len(matches) == 1, matches
    return np.asarray(data[matches[0]])


b_scale = max(np.max(np.abs(field(name))) for name in ("Bx", "By", "Bz"))
dx_min = np.min(
    np.asarray((ds.domain_right_edge - ds.domain_left_edge) / ds.domain_dimensions)
)
relative_divergence = np.max(np.abs(field("divB"))) / (b_scale / dx_min)
print(f"relative magnetic-divergence error: {relative_divergence:.6e}")
assert relative_divergence < 2.0e-12

material_values = {
    "sigma": 1.0,
    "epsilon": 8.8541878188e-12,
    "mu": 1.2566370612685e-6,
}
for name, expected in material_values.items():
    if any(item[1] == name for item in ds.field_list):
        relative_error = np.max(np.abs(field(name) - expected)) / expected
        print(f"relative {name} diagnostic error: {relative_error:.6e}")
        assert relative_error < 5.0e-6
