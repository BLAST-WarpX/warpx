#!/usr/bin/env python3
"""Refine actual native pressure/Faraday calls against analytic baroclinic curl."""

import os
import pathlib
import re
import subprocess
import sys

binary, inputs = map(pathlib.Path, sys.argv[1:3])
errors = []
for cells in (16, 32, 64):
    folder = pathlib.Path(f"spatial_{cells}")
    folder.mkdir(exist_ok=True)
    result = subprocess.run(
        [
            str(binary),
            str(inputs),
            f"amr.n_cell={cells} {cells}",
            "test.spatial_only=1",
            *sys.argv[3:],
        ],
        cwd=folder,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    match = re.search(r"spatial_relative_error=([\d.eE+-]+)", result.stdout)
    assert match, result.stdout
    errors.append(float(match[1]))
print(f"Native analytic spatial errors: {errors}")
assert errors[0] > 3.5 * errors[1]
assert errors[1] > 3.5 * errors[2]
