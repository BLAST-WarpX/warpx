#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Independent full-ray intersections and per-cell Beer-Lambert accounting."""

import argparse
import json
from pathlib import Path

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from read_raw_data import _read_buffer

parser = argparse.ArgumentParser()
add_precision_arguments(parser)
parser.add_argument("--position", nargs=3, type=float, required=True)
parser.add_argument("--direction", nargs=3, type=float, required=True)
parser.add_argument("--distance", type=float, required=True)
parser.add_argument("--periodic-z", action="store_true")
args = parser.parse_args()
_, _, cross_dtype = precision_dtypes(args)

# Assemble all analytic ray/surface intersections, then integrate intervals.
# This reference does not emulate the code's local cell-handoff decisions.
position = np.asarray(args.position)
direction = np.asarray(args.direction)
direction /= np.linalg.norm(direction)
x, y, z = position
nx, ny, nz = direction
events = [0.0, args.distance]
a, b = nx * nx + ny * ny, x * nx + y * ny
if a > 0:
    for radius in (0.25, 0.5, 0.75, 1.0):
        discriminant = b * b - a * (x * x + y * y - radius * radius)
        if discriminant > 0:
            events.extend((-b + sign * np.sqrt(discriminant)) / a for sign in (-1, 1))
if nz != 0:
    events.extend((face - z) / nz for face in np.arange(-1.0, 2.01, 0.25))
events = sorted(set(event for event in events if 0 <= event <= args.distance))
expected = np.zeros((4, 4))
remaining, escaped = 1.0, 0.0
for start, end in zip(events[:-1], events[1:]):
    if end - start < 8 * np.finfo(float).eps * args.distance:
        continue
    midpoint = position + 0.5 * (start + end) * direction
    radius = np.hypot(*midpoint[:2])
    axial = midpoint[2]
    if radius >= 1 or (not args.periodic_z and not 0 <= axial < 1):
        escaped, remaining = remaining, 0.0
        break
    i, j = int(radius / 0.25), int((axial % 1) / 0.25)
    removed = remaining * -np.expm1(-(1 + 2 * i + 3 * j) * (end - start))
    expected[i, j] += removed
    remaining -= removed
endpoint = position + args.distance * direction
axial_exit = not args.periodic_z and (
    endpoint[2] >= 1 or endpoint[2] < 0 or (endpoint[2] == 0 and nz < 0)
)
if np.hypot(*endpoint[:2]) >= 1 or axial_exit:
    escaped += remaining
    remaining = 0.0

plotfile = Path("diags/diag000001")
with (plotfile / "Header").open() as header:
    header.readline()
    names = [header.readline().strip() for _ in range(int(header.readline()))]
fields = _read_buffer(str(plotfile), str(plotfile / "Level_0/Cell_H"), names)
deposition = np.asarray(fields["radiation_material_energy"]).reshape(4, 4)
particle_energy = np.atleast_2d(np.loadtxt("diags/particle_energy.txt"))
initial = particle_energy[0, 2]
radiation = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
with Path("diags/radiation_energy.txt").open() as handle:
    columns = handle.readline().split()
loss_index = next(
    i for i, column in enumerate(columns) if "streaming_boundary_energy_loss" in column
)
# Some reduced-diagnostic headers prefix a separate '#' token.
if columns[0] == "#":
    loss_index -= 1
actual_escaped = radiation[-1, loss_index] / initial
report = {
    "expected_cell_absorption_fraction": expected.tolist(),
    "actual_cell_absorption_fraction": (deposition / initial).tolist(),
    "expected_remaining_fraction": remaining,
    "actual_remaining_fraction": float(particle_energy[-1, 2] / initial),
    "expected_escaped_fraction": escaped,
    "actual_escaped_fraction": float(actual_escaped),
}
Path("exact_rz_results.json").write_text(json.dumps(report, indent=2) + "\n")
rtol = 3.0e-5 if cross_dtype == np.float32 else 8.0e-13
atol = 2.0e-12 if cross_dtype == np.float64 else 3.0e-6
np.testing.assert_allclose(deposition / initial, expected, rtol=rtol, atol=atol)
np.testing.assert_allclose(
    particle_energy[-1, 2] / initial, remaining, rtol=rtol, atol=atol
)
np.testing.assert_allclose(actual_escaped, escaped, rtol=rtol, atol=atol)
np.testing.assert_allclose(
    np.sum(deposition) / initial + particle_energy[-1, 2] / initial + actual_escaped,
    1.0,
    rtol=rtol,
    atol=atol,
)
print("RZ full-ray attenuation, per-cell deposition and boundary inventory pass.")
