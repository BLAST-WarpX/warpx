#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL

"""Measure RZ PML boundary error against a larger vacuum domain."""

import numpy as np
import yt
from scipy.constants import c

yt.funcs.mylog.setLevel(0)
small = yt.load("diags/diag1000300")
reference = yt.load("../test_rz_pml_yee_reference/diags/diag1000300")
small.force_periodicity()
reference.force_periodicity()
np.testing.assert_allclose(
    small.domain_width / small.domain_dimensions,
    reference.domain_width / reference.domain_dimensions,
)
np.testing.assert_allclose(small.current_time, reference.current_time)
actual = small.covering_grid(0, small.domain_left_edge, small.domain_dimensions)
expected = reference.covering_grid(0, small.domain_left_edge, small.domain_dimensions)

# Normalize E and cB by the specified laser peak E=1 V/m. Test the modal fields
# separately so reconstruction at one azimuth cannot conceal a mode error.
errors = []
signals = []
for field in small.field_list:
    name = field[1]
    if field[0] != "boxlib" or name[0] not in "EB" or "_1_" not in name:
        continue
    scale = c if name.startswith("B") else 1.0
    values = actual[field].v
    target = expected[field].v
    assert np.isfinite(values).all() and np.isfinite(target).all()
    errors.append(float(np.max(np.abs(values - target)) * scale))
    signals.append(float(np.max(np.abs(target)) * scale))
assert len(errors) == 12  # real and imaginary parts of six field components
assert max(signals) > 0.01  # ensure this is a nontrivial pulse comparison
print(f"Maximum modal boundary error / laser peak: {max(errors)}")
assert max(errors) < 1.0e-3
