#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Run related load-balancing cases and their analysis as a single CTest."""

import subprocess
import sys
from pathlib import Path

group = sys.argv[1]
command = sys.argv[2:]
primary_input = f"inputs_test_2d_load_balance_{group}"
input_index = next(
    i for i, arg in enumerate(command) if Path(arg).name == primary_input
)
cases = {
    "split_merge": [
        ("refined_ratio4", primary_input),
        ("guard_cells", "inputs_base_2d_guard_cells"),
        ("merge", "inputs_base_2d_merge"),
    ],
    "costs": [("zero_cost", primary_input)],
}
analysis = Path(__file__).resolve().with_name("analysis.py")

for case, inputs in cases[group]:
    print(f"Running load-balancing case: {case}", flush=True)
    work = Path(case)
    work.mkdir(exist_ok=True)
    command[input_index] = inputs
    subprocess.run(command, cwd=work, check=True)
    if case == "merge":
        reference = work / "reference"
        reference.mkdir(exist_ok=True)
        subprocess.run(
            [*command, "algo.load_balance_intervals=0"], cwd=reference, check=True
        )
    subprocess.run([sys.executable, str(analysis), case], cwd=work, check=True)
