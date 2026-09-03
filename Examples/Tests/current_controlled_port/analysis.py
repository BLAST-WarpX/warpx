#!/usr/bin/env python3
"""Verify that the 3D port projection preserves the magnetic constraint."""

import glob

import numpy as np
import yt

yt.funcs.mylog.setLevel(50)

plotfiles = sorted(glob.glob("diags/diag1*"))
assert plotfiles
ds = yt.load(plotfiles[-1])
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
