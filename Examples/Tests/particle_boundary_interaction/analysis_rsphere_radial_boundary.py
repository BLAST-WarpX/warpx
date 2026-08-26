#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check RSPHERE radial reflecting and thermal particle boundaries."""

import sys
from pathlib import Path

import numpy as np
import yt
from scipy.constants import c, m_e

yt.funcs.mylog.setLevel(0)


def particle_state(plotfile):
    dataset = yt.load(str(plotfile))
    data = dataset.all_data()
    particle_id = np.asarray(data["test", "particle_id"], dtype=np.int64)
    order = np.argsort(particle_id)
    radius = np.asarray(data["test", "particle_position_x"])[order]
    theta = np.asarray(data["test", "particle_theta"])[order]
    phi = np.asarray(data["test", "particle_phi"])[order]
    position = radius[:, None] * np.column_stack(
        [
            np.cos(theta) * np.cos(phi),
            np.sin(theta) * np.cos(phi),
            np.sin(phi),
        ]
    )
    normalized_momentum = np.column_stack(
        [
            np.asarray(data["test", f"particle_momentum_{axis}"])[order] / (m_e * c)
            for axis in "xyz"
        ]
    )
    return particle_id[order], position, normalized_momentum


final_plotfile = Path(sys.argv[1])
boundary_type = sys.argv[2]
initial_plotfile = final_plotfile.with_name("diag000000")

initial_id, initial_position, initial_u = particle_state(initial_plotfile)
final_id, final_position, final_u = particle_state(final_plotfile)

assert np.array_equal(final_id, initial_id), "the radial boundary changed particle IDs"
assert initial_id.size >= 32, "the test needs angular coverage"

initial_er = initial_position / np.linalg.norm(initial_position, axis=1)[:, None]
final_er = final_position / np.linalg.norm(final_position, axis=1)[:, None]
initial_ur = np.einsum("ij,ij->i", initial_u, initial_er)
final_ur = np.einsum("ij,ij->i", final_u, final_er)

np.testing.assert_allclose(initial_er, final_er, rtol=2.0e-14, atol=2.0e-14)
np.testing.assert_allclose(initial_ur, 1.0, rtol=2.0e-14, atol=2.0e-14)

if boundary_type == "reflecting":
    np.testing.assert_allclose(final_u, -initial_u, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(final_ur, -1.0, rtol=2.0e-14, atol=2.0e-14)
elif boundary_type == "thermal":
    assert np.all(final_ur < 0.0), (
        "thermal radial re-emission must point inward; "
        f"{np.count_nonzero(final_ur >= 0.0)} of {final_ur.size} particles point outward"
    )
    assert np.all(np.linalg.norm(final_u, axis=1) > 0.0)
else:
    raise ValueError(f"unknown boundary type {boundary_type!r}")

print(
    f"PASS: RSPHERE {boundary_type} radial boundary; "
    f"ur range [{final_ur.min():.6e}, {final_ur.max():.6e}]"
)
