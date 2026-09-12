#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Separate mesh/time checks against a moving reflecting-cavity diffusion mode.

This is prescribed-material radiation accuracy, not native particle-wall
qualification. The analytic reference uses the gray diffusion limit, not a
FLASH fluid closure. --production runs the finer study and saves a plot.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np


def main(executable, production=False):
    executable = str(Path(executable).resolve())
    meshes = (32, 64, 128) if production else (16, 32, 64)
    spatial_steps = 6400 if production else 1600
    time_steps = (200, 400, 800) if production else (100, 200, 400)
    cases = {}
    configurations = [(n, spatial_steps) for n in meshes]
    configurations += [(meshes[-1], steps) for steps in time_steps]
    for cells, steps in configurations:
        command = [
            executable,
            "test.mode=reflecting",
            "test.beta=0.002",
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
        log = Path(f"cavity-n{cells}-s{steps}.log")
        log.write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError(f"Cavity simulation failed; see {log.resolve()}")
        match = re.search(
            r"relative Fourier error=(\S+) real=(\S+) imag=(\S+).*"
            r"energy=(\S+) momentum=(\S+) profile_L1=(\S+)",
            completed.stdout,
        )
        assert match, completed.stdout
        values = list(map(float, match.groups()))
        assert np.isfinite(values).all()
        error, amplitude, imaginary, energy, momentum, profile = values
        assert imaginary == 0  # A Neumann cosine mode, not a traveling Fourier mode.
        assert energy < 1e-10 and momentum < 1e-10
        cases[(cells, steps)] = {
            "cells": cells,
            "steps": steps,
            "relative_amplitude_error": error,
            "amplitude": amplitude,
            "profile_L1_over_amplitude": profile,
            "energy_residual": energy,
            "momentum_residual": momentum,
        }
        print(
            f"Cavity n={cells} steps={steps}: amplitude={error:.8g}, L1={profile:.8g}",
            flush=True,
        )

    spatial = [cases[(n, spatial_steps)] for n in meshes]
    ratios = {}
    for name in ("relative_amplitude_error", "profile_L1_over_amplitude"):
        errors = np.array([row[name] for row in spatial])
        ratios[name] = (errors[:-1] / errors[1:]).tolist()
        assert min(ratios[name]) > 2, (name, errors)
        assert errors[-1] < (5e-5 if production else 1e-3), (name, errors)
    amplitudes = np.array(
        [cases[(meshes[-1], steps)]["amplitude"] for steps in time_steps]
    )
    differences = abs(np.diff(amplitudes))
    assert np.all(differences > 0)
    temporal_ratio = float(differences[0] / differences[1])
    assert 1.8 < temporal_ratio < 2.2, temporal_ratio
    report = {
        "production_resolution": production,
        "scope": "gray radiation in prescribed tangentially moving material; fixed optical mirrors",
        "reference": "R=1+0.1*exp(-pi^2*c*t/(3*kappa*gamma))*cos(pi*x/L), L=1",
        "beta": 0.002,
        "kappa_times_L": 10000,
        "c_times_duration_over_L": 1000,
        "cases": list(cases.values()),
        "spatial_ratios": ratios,
        "temporal_ratio": temporal_ratio,
    }
    Path("reflecting_cavity_results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if production:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for name, label in (
            ("relative_amplitude_error", "Amplitude error"),
            ("profile_L1_over_amplitude", "Full-profile L1 error / amplitude"),
        ):
            axes[0].loglog(meshes, [row[name] for row in spatial], "o-", label=label)
        axes[0].set(
            xlabel="Cells (6400 fixed steps)", ylabel="Error against diffusion limit"
        )
        axes[0].legend()
        axes[1].loglog(time_steps[:-1], differences, "o-")
        axes[1].set(
            xlabel="Coarse step count (128 fixed cells)",
            ylabel="Amplitude self-difference",
        )
        axes[1].set_title(f"Temporal ratio: {temporal_ratio:.4f}")
        fig.suptitle(
            "Reflecting cavity: gray radiation, tangential material drift beta=0.002"
        )
        fig.savefig("reflecting_cavity_refinement.png", dpi=160)
        plt.close(fig)
    print(f"Reflecting-cavity refinement passed; temporal ratio={temporal_ratio:.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable")
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()
    main(args.executable, args.production)
