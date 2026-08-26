#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import math
from pathlib import Path


def read_position(diag_name):
    path = Path("diags/reducedfiles") / f"{diag_name}.txt"
    rows = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    row = [float(value) for value in rows[-1].split()]

    # ParticleExtrema columns are step, time, xmin, xmax, ymin, ymax, zmin, zmax, ...
    position_min = tuple(row[index] for index in (2, 4, 6))
    position_max = tuple(row[index] for index in (3, 5, 7))
    for minimum, maximum in zip(position_min, position_max):
        assert math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1.0e-15)
    return position_min


expected_positions = {
    "radial_position": (0.5, 0.0, 0.0),
    "off_axis_position": (0.3, 0.4, -1.2),
}

for diagnostic, expected in expected_positions.items():
    actual = read_position(diagnostic)
    print(f"{diagnostic}: expected {expected}, actual {actual}")
    for component, value, reference in zip("xyz", actual, expected):
        assert math.isclose(value, reference, rel_tol=1.0e-14, abs_tol=1.0e-15), (
            f"{diagnostic} {component}: expected {reference}, got {value}"
        )
