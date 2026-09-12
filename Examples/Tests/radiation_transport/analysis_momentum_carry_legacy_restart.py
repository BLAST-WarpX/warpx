#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Verify that a schema-v1 restart initializes absent momentum carries to zero."""

from pathlib import Path

import numpy as np

path = Path("diags/radiation_momentum.txt")
with path.open() as stream:
    labels = [
        token.split("]", 1)[1]
        for token in stream.readline().lstrip("#").split()
        if token.startswith("[") and "]" in token
    ]
data = np.atleast_2d(np.loadtxt(path))
assert data.shape[1] == len(labels)

step_column = labels.index("step()")
repeated_run = np.flatnonzero(np.diff(data[:, step_column]) <= 0)
if repeated_run.size:
    data = data[repeated_run[-1] + 1 :]
assert data.shape[0] == 1
assert int(data[0, step_column]) == 149

pending_labels = [
    f"pending_{source}_material_{axis}(kg*m/s)"
    for source in ("streaming", "diffusion")
    for axis in ("x", "y", "z")
]
assert set(pending_labels) <= set(labels)
pending = np.array([data[0, labels.index(label)] for label in pending_labels])
np.testing.assert_array_equal(pending, 0.0)

print("legacy checkpoint initialized all absent radiation momentum carries to zero")
