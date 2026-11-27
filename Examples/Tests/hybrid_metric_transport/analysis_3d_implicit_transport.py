#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Native oblique transport/reflecting-wall checks, not a total-PdV energy claim."""

import argparse
import sys
from pathlib import Path

import numpy as np
import yt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "radiation_transport"))
from analysis_moving_moment_interface import nodes  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path)
args = parser.parse_args()
yt.set_log_level(50)
cells = np.array([16, 16, 16])
lower = np.array([-0.21e-3, -0.13e-3, -0.07e-3])
upper = np.array([0.59e-3, 0.57e-3, 0.83e-3])
width = upper - lower
periodic = [True, True, False]
volume = np.full((16, 16, 17), np.prod(width / cells), dtype=np.longdouble)
volume[..., 0] *= 0.5
volume[..., -1] *= 0.5


def load(directory, step):
    plot = Path(directory) / f"diags/plt{step:06d}"
    dataset = yt.load(str(plot))
    data = dataset.all_data()
    ids = np.asarray(data["ions", "particle_id"].v)
    cpu = np.asarray(data["ions", "particle_cpu"].v)
    order = np.lexsort((cpu, ids))
    particles = {
        name: np.asarray(data["ions", name].v)[order]
        for name in ["particle_id", "particle_cpu", "particle_weight"]
        + [f"particle_position_{d}" for d in "xyz"]
        + [f"particle_momentum_{d}" for d in "xyz"]
    }
    rho = nodes(plot / "raw_fields/Level_0/rho_fp_H", cells, periodic)
    temperature = nodes(
        Path(directory) / f"diags/chk{step:06d}/Level_0/"
        "hybrid_electron_temperature_fp[level=0]_H",
        cells,
        periodic,
    )
    assert np.all(rho >= 0) and np.all(np.isfinite(temperature))
    assert np.min(temperature[rho > 0]) > 0
    for values in particles.values():
        assert np.all(np.isfinite(values))
    return float(dataset.current_time), particles, rho, temperature


_, initial, rho0, _ = load(args.reference or Path("."), 0)
time, final, rho, temperature = load(Path("."), 40)
assert abs(time / 4.0e-9 - 1) < 1.0e-12
for key in ("particle_id", "particle_cpu", "particle_weight"):
    np.testing.assert_array_equal(final[key], initial[key])
charge = np.longdouble("1.602176634e-19") * np.sum(
    initial["particle_weight"], dtype=np.longdouble
)
for values in (rho0, rho):
    np.testing.assert_allclose(np.sum(values * volume), charge, rtol=1.0e-12, atol=0)
    assert np.count_nonzero(values == 0) > values.size // 4

normalized_u = np.array([3.0e-4, 4.0e-4, 1.0e-3])
velocity = 299792458.0 * normalized_u / np.sqrt(1 + np.sum(normalized_u**2))
for axis, label in enumerate("xyz"):
    unfolded = (
        initial[f"particle_position_{label}"] - lower[axis] + velocity[axis] * time
    )
    if axis < 2:
        expected = lower[axis] + unfolded % width[axis]
        sign = np.ones_like(expected)
    else:
        phase = unfolded % (2 * width[axis])
        expected = lower[axis] + np.where(
            phase < width[axis], phase, 2 * width[axis] - phase
        )
        sign = np.where(phase < width[axis], 1.0, -1.0)
        assert np.all(sign == -1), "The trajectory must actually exercise z reflection."
    np.testing.assert_allclose(
        final[f"particle_position_{label}"],
        expected,
        rtol=0,
        atol=512 * np.finfo(float).eps * width[axis],
    )
    np.testing.assert_allclose(
        final[f"particle_momentum_{label}"],
        sign * initial[f"particle_momentum_{label}"],
        rtol=128 * np.finfo(float).eps,
        atol=0,
    )

if args.reference:
    _, reference, reference_rho, reference_temperature = load(args.reference, 40)
    np.testing.assert_allclose(
        rho,
        reference_rho,
        rtol=0,
        atol=2048 * np.finfo(float).eps * np.max(reference_rho),
    )
    np.testing.assert_allclose(
        temperature,
        reference_temperature,
        rtol=0,
        atol=2048 * np.finfo(float).eps * np.max(reference_temperature),
    )
    for key in reference:
        np.testing.assert_array_equal(final[key], reference[key])
    print("Native oblique 3D restart comparison passes.")
print(
    "Native oblique 3D transport: charge, positive thermal state and wall trajectory checks pass."
)
