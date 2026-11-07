#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from conducting_cylinder_solution import (
    ProblemParameters,
    analytic_solution,
    compute_error_metrics,
    load_plotfile,
    make_error_masks,
)
from plot_conducting_cylinder import DPI, PLOT_PREFIX, default_output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot resolution-dependent errors for the conducting-cylinder test."
    )
    parser.add_argument("plotfiles", nargs="+", help="WarpX plotfiles to compare")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for the plot; default is the first plotfile's <run-dir>/plots",
    )
    parser.add_argument(
        "--output-name",
        default=f"{PLOT_PREFIX}_resolution_errors.png",
        help="output PNG filename",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.plotfiles[0])
    output_dir.mkdir(parents=True, exist_ok=True)

    params = ProblemParameters()
    rows = []
    for plotfile in args.plotfiles:
        data = load_plotfile(plotfile)
        exact = analytic_solution(data, params)
        masks = make_error_masks(data, params)
        metrics = compute_error_metrics(data, exact, masks)
        rows.append((min(data.dx[0], data.dx[1]), data.dims[0], metrics))

    rows.sort(key=lambda row: row[0], reverse=True)
    dx = [row[0] for row in rows]
    nx = [row[1] for row in rows]
    field_error = [row[2].field_rms_relative_error for row in rows]
    shell_error = [row[2].shell_field_rms_relative_error for row in rows]
    phi_error = [row[2].phi_rms_relative_error for row in rows]

    for cell_size, cells, metrics in rows:
        print(
            f"nx={cells:5d} dx={cell_size:.8e} "
            f"field={metrics.field_rms_relative_error:.8e} "
            f"shell={metrics.shell_field_rms_relative_error:.8e} "
            f"phi={metrics.phi_rms_relative_error:.8e}"
        )

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.6), constrained_layout=True)
    ax.loglog(dx, field_error, "o-", label=r"$E$ RMS")
    ax.loglog(dx, shell_error, "s-", label=r"shell $E$ RMS")
    ax.loglog(dx, phi_error, "^-", label=r"$\phi$ RMS")

    for cell_size, cells, error in zip(dx, nx, phi_error):
        ax.annotate(str(cells), (cell_size, error))

    ax.invert_xaxis()
    ax.set_xlabel("cell size")
    ax.set_ylabel("relative RMS error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    path = output_dir / args.output_name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
