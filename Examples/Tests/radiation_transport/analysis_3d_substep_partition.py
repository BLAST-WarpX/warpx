#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Compare complete interval histories after changed-rank particle restart.

Subsegment optical-depth carry is transactional within one outer interval, not
checkpoint state. Restart here occurs at its boundary, before replay starts.
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("executable")
parser.add_argument("inputs")
parser.add_argument("fine_inputs")
parser.add_argument("--mpiexec", required=True)
parser.add_argument("--numproc", default="-n")
args = parser.parse_args()
reference = None
checkpoint = str(Path("rank_change_checkpoint").resolve())
for ranks, inputs, label in (
    (1, args.inputs, "reference"),
    (2, args.inputs, "restart2"),
    (3, args.fine_inputs, "restart3_fine"),
):
    command = [args.mpiexec, args.numproc, str(ranks), args.executable, inputs]
    option = "checkpoint_to" if reference is None else "restart_from"
    command += [f"substep_test.{option}={checkpoint}"]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    Path(f"{label}.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-6000:]
    histories = {}
    for line in result.stdout.splitlines():
        if line.startswith("PARTITION_HISTORY "):
            _, name, *values = line.split()
            histories[name] = (
                [int(v) for v in values[:5]],
                np.asarray(values[5:], dtype=np.longdouble),
            )
    assert set(histories) == {"flight", "scatter_before", "scatter_after"}
    if reference is None:
        reference = histories
    else:
        for name, (counters, values) in histories.items():
            assert counters == reference[name][0], (label, name, counters)
            np.testing.assert_allclose(
                values,
                reference[name][1],
                rtol=5e-12,
                atol=0,
                err_msg=f"{label}: {name}",
            )
print("3D interval histories match after MPI1->MPI2/MPI3 restart and layout change")
