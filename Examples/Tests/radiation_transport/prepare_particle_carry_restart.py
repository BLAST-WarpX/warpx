#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Archive this regression's previous restart output before a fresh replay."""

import argparse
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("case", type=Path)
args = parser.parse_args()
case = args.case.resolve()
assert case.name in {
    "test_1d_radiation_particle_carry_restart",
    "test_2d_radiation_particle_carry_restart",
    "test_1d_radiation_transport_moving_beam_restart",
}
output = case / "diags"
assert not output.is_symlink()
if output.exists():
    archive = case / "previous_runs"
    assert not archive.is_symlink()
    archive.mkdir(exist_ok=True)
    target = archive / str(time.time_ns())
    output.rename(target)
    print(f"Preserved previous restart output at {target}")
