#!/usr/bin/env python3
#
# The fluid ion response (implicit_evolve.use_fluid_ion_response) only changes the Jacobian of
# the Newton solve, not the solution it converges to: the final fields of this run must match
# those of the particle-push run of the same deck (--ref) to within the Newton tolerance,
# relative to the field perturbation. Newton must also converge (the deck requires it); with
# 64 particles per cell the grid-moment response omits the particle noise of the exact one,
# which costs a few iterations at this tight tolerance, so the count is only bounded loosely.

import argparse

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ref", required=True, help="run directory of the particle-push run"
)
parser.add_argument("--rtol", type=float, default=1.0e-6)
args = parser.parse_args()

ts = OpenPMDTimeSeries("diags/field_diags")
tr = OpenPMDTimeSeries(f"{args.ref}/diags/field_diags")
it = ts.iterations[-1]
assert it == tr.iterations[-1], "the two runs must end at the same step"

worst = 0.0
for field in ("B", "E", "j"):
    for comp in ("x", "y", "z"):
        a, _ = ts.get_field(field, comp, iteration=it)
        b, _ = tr.get_field(field, comp, iteration=it)
        # relative to the perturbation (the uniform B0 along z is removed)
        scale = (
            np.max(np.abs(b - np.mean(b)))
            if (field, comp) == ("B", "z")
            else np.max(np.abs(b))
        )
        err = np.max(np.abs(a - b)) / scale
        print(f"{field}{comp}: max |fluid - particle| / perturbation = {err:.3e}")
        worst = max(worst, err)
assert worst < args.rtol, (
    f"fluid and particle-push solutions differ by {worst:.3e} > {args.rtol}"
)


def newton_per_step(path):
    rows = np.atleast_2d(np.loadtxt(path))
    return rows[:, 2].mean()


n_fluid = newton_per_step("newton.txt")
n_ref = newton_per_step(f"{args.ref}/newton.txt")
print(f"Newton iterations per step: fluid {n_fluid:.2f}, particle push {n_ref:.2f}")
assert n_fluid <= 3.0 * n_ref, (
    "the fluid ion response converges far slower than expected"
)
