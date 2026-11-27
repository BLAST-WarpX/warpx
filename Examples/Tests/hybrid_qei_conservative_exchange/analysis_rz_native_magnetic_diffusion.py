#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Check native RZ HybridPIC evolution of a diffusing magnetic perturbation."""

import argparse
from pathlib import Path

import numpy as np
import yt

FINAL_TIME = 0.05
DIFFUSIVITY = 0.1
B_BACKGROUND = 0.1
B_PERTURBATION = 0.01
SIGMA2 = 0.04


def final_plotfile(run_dir):
    diags = run_dir / "diags"
    candidates = [
        path
        for path in diags.glob("plt*")
        if path.name.removeprefix("plt").isdigit() and (path / "Header").is_file()
    ]
    assert candidates, f"No plotfiles found in {diags}"
    return max(candidates, key=lambda path: int(path.name.removeprefix("plt")))


def plot_field(covering_grid, dataset, name):
    fields = [field for field in dataset.field_list if field[1] == name]
    assert len(fields) == 1, f"Expected exactly one {name!r} field, found {fields}"
    return np.squeeze(np.asarray(covering_grid[fields[0]], dtype=np.float64))


def check(plotfile):
    dataset = yt.load(str(plotfile))
    dimensions = [int(value) for value in dataset.domain_dimensions]
    n_r, n_z = dimensions[:2]
    covering_grid = dataset.covering_grid(
        level=0,
        left_edge=dataset.domain_left_edge,
        dims=dataset.domain_dimensions,
    )
    bz = plot_field(covering_grid, dataset, "Bz")
    rho = plot_field(covering_grid, dataset, "rho")
    assert bz.shape == rho.shape == (n_r, n_z)
    assert np.all(np.isfinite(bz)) and np.all(np.isfinite(rho))
    assert np.all(rho > 0)

    time = float(getattr(dataset.current_time, "d", dataset.current_time))
    np.testing.assert_allclose(time, FINAL_TIME, rtol=1e-12, atol=0.0)

    left = float(dataset.domain_left_edge[0].d)
    right = float(dataset.domain_right_edge[0].d)
    dr = (right - left) / n_r
    width = SIGMA2 + 4.0 * DIFFUSIVITY * time
    face_r = left + np.arange(n_r + 1) * dr
    face_bz = B_BACKGROUND + (
        B_PERTURBATION * SIGMA2 / width * np.exp(-(face_r * face_r) / width)
    )
    # WarpX plotfiles cell-center fully nodal fields by averaging adjacent nodes.
    expected_bz = (0.5 * (face_bz[:-1] + face_bz[1:]))[:, None]

    error = bz - expected_bz
    max_relative_linf = np.max(np.abs(error)) / B_PERTURBATION
    cell_r = left + (np.arange(n_r) + 0.5) * dr
    weights = np.broadcast_to(cell_r[:, None], error.shape)
    weighted_relative_l2 = (
        np.sqrt(np.average(error * error, weights=weights)) / B_PERTURBATION
    )
    assert max_relative_linf < 0.01, max_relative_linf

    print(
        f"RZ native magnetic diffusion: time={time:.12g}s, grid={n_r}x{n_z}, "
        f"max_relative_linf={max_relative_linf:.6e}, "
        f"weighted_relative_l2={weighted_relative_l2:.6e}"
    )
    return n_r, float(max_relative_linf), float(weighted_relative_l2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plotfile", nargs="?", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--coarse-reference", type=Path)
    arguments = parser.parse_args()
    if (arguments.plotfile is None) == (arguments.run_dir is None):
        parser.error("provide exactly one of PLOTFILE or --run-dir")
    selected = arguments.plotfile or final_plotfile(arguments.run_dir)
    fine = check(selected)
    if arguments.coarse_reference is not None:
        coarse = check(final_plotfile(arguments.coarse_reference))
        assert fine[0] == 2 * coarse[0], (fine, coarse)
        assert 3.0 * fine[1] < coarse[1], (fine, coarse)
        assert 3.0 * fine[2] < coarse[2], (fine, coarse)
        print(
            f"Refinement ratios: Linf={coarse[1] / fine[1]:.6g}, L2={coarse[2] / fine[2]:.6g}"
        )
