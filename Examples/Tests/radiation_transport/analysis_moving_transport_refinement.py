#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Separate spatial and temporal moving-diffusion refinement, plus beam transport."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def main(executable):
    executable = str(Path(executable).resolve())
    cases = {}
    for mode, cells, steps in [
        ("trapped", 64, 400),
        ("trapped", 128, 400),
        ("trapped", 256, 400),
        ("trapped", 256, 200),
        ("trapped", 256, 100),
        ("beam", 64, 100),
        ("beam", 128, 200),
        ("beam", 256, 400),
    ]:
        command = [
            executable,
            f"test.mode={mode}",
            f"test.cells={cells}",
            f"test.steps={steps}",
            "amrex.the_arena_init_size=0",
        ]
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"Failed {' '.join(command)}\n{completed.stdout}")
        match = re.search(
            r"relative Fourier error=(\S+) real=(\S+) imag=(\S+)", completed.stdout
        )
        assert match, completed.stdout
        values = list(map(float, match.groups()))
        assert all(abs(value) < float("inf") for value in values)
        cases[(mode, cells, steps)] = values
        print(f"{mode} cells={cells} steps={steps}: error={values[0]:.8g}", flush=True)

    spatial = [cases[("trapped", n, 400)][0] for n in (64, 128, 256)]
    assert spatial[-1] < 0.035, spatial
    assert all(a / b > 1.6 for a, b in zip(spatial, spatial[1:])), spatial
    temporal = [complex(*cases[("trapped", 256, n)][1:]) for n in (100, 200, 400)]
    ratio = abs(temporal[0] - temporal[1]) / abs(temporal[1] - temporal[2])
    assert 1.6 < ratio < 2.4, ratio
    beam = [cases[("beam", n, s)][0] for n, s in ((64, 100), (128, 200), (256, 400))]
    assert beam[-1] < 0.01, beam
    assert all(a / b > 1.6 for a, b in zip(beam, beam[1:])), beam
    report = {
        "cases": [
            {
                "mode": key[0],
                "cells": key[1],
                "steps": key[2],
                "relative_error": value[0],
                "real": value[1],
                "imag": value[2],
            }
            for key, value in cases.items()
        ],
        "temporal_ratio": ratio,
    }
    Path("moving_transport_refinement.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"Moving transport refinement passed; temporal ratio={ratio:.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("executable")
    main(parser.parse_args().executable)
