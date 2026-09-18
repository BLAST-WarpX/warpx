#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check fields, particles, layouts, and efficiency after runtime load balancing."""

import math
import sys
from pathlib import Path

case = sys.argv[1]
path = Path("diags/reducedfiles")


def read_diagnostic(name, directory=path):
    lines = (directory / f"{name}.txt").read_text().splitlines()
    return lines[0].split(), [line.split() for line in lines[1:] if line.strip()]


_, energy = read_diagnostic("FE")
assert [int(row[0]) for row in energy] == list(range(5))
if case == "merge":
    assert all(float(row[2]) > 0.0 for row in energy), energy
    for name in ("FE", "FP"):
        header, values = read_diagnostic(name)
        ref_header, reference = read_diagnostic(name, Path("reference") / path)
        assert header == ref_header
        assert len(values) == len(reference)
        assert {int(row[0]) for row in values} == set(range(5))
        assert all(len(row) == len(header) for row in values + reference)
        # Compare each field component on its own scale, including near-zero values.
        for column in range(len(header)):
            expected = [float(row[column]) for row in reference]
            scale = max(abs(value) for value in expected)
            for row, value in zip(values, expected):
                assert math.isclose(
                    float(row[column]), value, rel_tol=1.0e-11, abs_tol=1.0e-11 * scale
                ), (name, row, column, value)
else:
    for row in energy:
        # No initial fields or particle velocities: these must remain exactly zero.
        assert all(float(value) == 0.0 for value in row[2:]), row

_, efficiency = read_diagnostic("LBE")
assert [int(row[0]) for row in efficiency] == list(range(5))
for row in efficiency:
    if int(row[0]) >= 2:
        assert all(0.0 < float(value) <= 1.0 for value in row[2:]), row

_, particles = read_diagnostic("PN")
assert [int(row[0]) for row in particles] == list(range(5))
particle_counts = [int(float(row[2])) for row in particles]
assert particle_counts[0] > 0
assert all(count == particle_counts[0] for count in particle_counts), particle_counts

header, costs = read_diagnostic("LBC")
# Locate each box using the header, allowing the extra GPU ID column on GPUs
# and NaN padding when the number of boxes changes during the run.
box_columns = [i for i, name in enumerate(header) if "cost_box_" in name]
layouts = {}
for row in costs:
    boxes = []
    for start in box_columns:
        if math.isfinite(float(row[start])):
            boxes.append(
                {
                    "cost": float(row[start]),
                    "level": int(float(row[start + 2])),
                    "lo": (int(float(row[start + 3])), int(float(row[start + 4]))),
                    "cells": int(float(row[start + 6])),
                    "particles": int(float(row[start + 7])),
                }
            )
    layouts[int(row[0])] = boxes

assert sorted(layouts) == list(range(5)), layouts.keys()

before, after = layouts[1], layouts[2]
assert len(layouts[4]) == len(after), (
    "The second load balance changed the settled layout"
)

if case == "merge":
    assert len(after) < len(before), (before, after)
elif case == "zero_cost":
    old_coarse = [box for box in before if box["level"] == 0]
    new_coarse = [box for box in after if box["level"] == 0]
    # Split the particle-heavy coarse box while retaining its empty neighbor.
    assert len(old_coarse) == 2
    assert {box["lo"] for box in new_coarse} == {(0, 0), (0, 8), (16, 0)}, new_coarse
    assert all(box["particles"] > 0 for box in new_coarse if box["lo"][0] == 0)
    assert all(box["particles"] == 0 for box in new_coarse if box["lo"][0] == 16)

    # The particle-free refined level must keep its layout.
    old_boxes = [box for box in before if box["level"] == 1]
    new_boxes = [box for box in after if box["level"] == 1]
    assert old_boxes
    assert all(box["cost"] == 0.0 and box["particles"] == 0 for box in old_boxes)
    assert new_boxes == old_boxes, (old_boxes, new_boxes)
elif case == "refined_ratio4":
    old_boxes = [box for box in before if box["level"] == 1]
    new_boxes = [box for box in after if box["level"] == 1]
    assert old_boxes
    assert len(new_boxes) > len(old_boxes), (old_boxes, new_boxes)
    # A split at a non-multiple of 4 gives neighboring boxes overlapping coarse cells.
    assert all(all(index % 4 == 0 for index in box["lo"]) for box in new_boxes)
elif case == "guard_cells":
    # Rho, whose ghosts exceed those of E/B and J, sets the strict lower bound.
    # Coarse patches have the same ghost widths but half as many valid cells.
    for lev, (nx, nz) in enumerate(((16, 8), (32, 16))):
        boxes = [box for box in after if box["level"] == lev]
        extent = 32 * 2**lev
        assert {box["lo"] for box in boxes} == {
            (x, z) for x in range(0, extent, nx) for z in range(0, extent, nz)
        }, boxes
        # LoadBalanceCosts counts points of Ex, which is nodal along z.
        assert all(box["cells"] == nx * (nz + 1) for box in boxes), boxes
        assert len(boxes) > sum(box["level"] == lev for box in before)
else:
    raise ValueError(f"Unknown load-balance case: {case}")

print(f"{case}: fields, particles, layout, and efficiency checks passed")
