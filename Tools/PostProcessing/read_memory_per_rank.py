#!/usr/bin/env python3
"""
Read and visualize WarpX ``MemoryPerRank`` reduced-diagnostic output.

Each participating MPI rank writes one file named
``<path>/<file_prefix>.<zero-padded rank>.yaml`` containing a stream of YAML
documents (one per diagnostic interval). This helper loads all rank files
from a directory into a tidy ``pandas.DataFrame`` and optionally produces
summary plots of total arena usage and host-process memory over time.

Requires: pyyaml, pandas, matplotlib.

Usage (as a library)::

    import read_memory_per_rank as mpr
    df = mpr.load("diags/reducedfiles/MemoryPerRank/", prefix="MPR")
    fig = mpr.plot(df)

Usage (as a script)::

    python3 read_memory_per_rank.py diags/reducedfiles/MemoryPerRank/ \\
        --prefix MPR --output mpr_timeline.png
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any, Iterable

import pandas as pd
import yaml


# Columns we want to pull up to top-level for easy plotting. All MemoryPerRank
# memory values are emitted as float MB with 3-decimal precision.
_PROCESS_KEYS = ["vm_peak_mb", "vm_size_mb", "vm_hwm_mb", "vm_rss_mb"]
# Arenas that AMReX may or may not emit as a distinct block. Missing arenas
# are simply not present in the DataFrame column list.
_ARENA_KEYS = ["main", "device", "managed", "pinned", "comms"]


def _flatten_snapshot(snap: dict[str, Any], rank_file: str) -> dict[str, Any]:
    """Flatten one YAML snapshot into a single-row dict suitable for a DataFrame."""
    row: dict[str, Any] = {
        "file": os.path.basename(rank_file),
        "step": snap.get("step"),
        "time": snap.get("time"),
    }

    mpi = snap.get("mpi", {}) or {}
    row["rank"] = mpi.get("rank")
    row["nprocs"] = mpi.get("size")

    host = snap.get("host", {}) or {}
    row["hostname"] = host.get("name")

    gpu = snap.get("gpu", {}) or {}
    row["gpu_device_id"] = gpu.get("device_id")
    row["gpu_total_mb"] = gpu.get("total_mb")
    row["gpu_free_mb"] = gpu.get("free_mb")

    arenas = snap.get("arenas", {}) or {}
    for name in _ARENA_KEYS:
        a = arenas.get(name)
        if isinstance(a, dict):
            row[f"arena_{name}_allocated_mb"] = a.get("allocated_mb")
            row[f"arena_{name}_used_mb"] = a.get("used_mb")

    proc = snap.get("process", {}) or {}
    for k in _PROCESS_KEYS:
        row[k] = proc.get(k)

    return row


def _iter_rank_files(directory: str, prefix: str) -> Iterable[str]:
    """All ``<directory>/<prefix>.*.yaml`` files in numeric-rank order."""
    pattern = os.path.join(directory, f"{prefix}.*.yaml")
    files = glob.glob(pattern)
    # Filenames are zero-padded, so a lexicographic sort is rank-order too.
    files.sort()
    return files


def load(directory: str, prefix: str = "MPR") -> pd.DataFrame:
    """Load all ``MemoryPerRank`` YAML files in *directory* into a DataFrame.

    Parameters
    ----------
    directory
        Path to the directory containing the per-rank files, typically
        ``diags/reducedfiles/<reduced_diags_name>/``.
    prefix
        File prefix used by the diagnostic, matching ``<rd>.file_prefix``
        (default ``"MPR"``).

    Returns
    -------
    pandas.DataFrame
        One row per (rank, step). Columns include ``step``, ``time``, ``rank``,
        ``hostname``, ``gpu_device_id``, ``gpu_total_mb``, ``gpu_free_mb``,
        ``arena_<name>_allocated_mb`` / ``arena_<name>_used_mb`` for each arena
        reported, and ``vm_peak_mb`` / ``vm_size_mb`` / ``vm_hwm_mb`` /
        ``vm_rss_mb`` (Linux only). All memory values are in MB.
    """
    rows: list[dict[str, Any]] = []
    n_files = 0
    for f in _iter_rank_files(directory, prefix):
        n_files += 1
        with open(f) as fh:
            for snap in yaml.safe_load_all(fh):
                if snap is None:
                    # Empty document (can happen if a file ends with `---\n`).
                    continue
                rows.append(_flatten_snapshot(snap, f))

    if n_files == 0:
        raise FileNotFoundError(
            f"No files match {os.path.join(directory, prefix)}.*.yaml; "
            "check the directory and prefix."
        )
    if not rows:
        raise ValueError(
            f"Found {n_files} MemoryPerRank files in {directory} but no YAML "
            "documents inside them; was the diagnostic ever triggered?"
        )

    df = pd.DataFrame(rows)
    # Stable ordering helps anyone iterating per-rank time series.
    df = df.sort_values(["rank", "step"]).reset_index(drop=True)
    return df


def plot(
    df: pd.DataFrame,
    output: str | None = None,
    show: bool = False,
):
    """Plot ``MemoryPerRank`` time series for all ranks.

    Creates two stacked panels:

    * Top: per-rank total AMReX arena memory *allocated* (MB), with one curve
      per rank. This is the device/host memory that AMReX explicitly reserved.
    * Bottom: per-rank host-process resident memory (``VmRSS`` in MB) and its
      high-water mark (``VmHWM``). This captures everything including things
      AMReX arenas don't track.

    Parameters
    ----------
    df
        DataFrame as returned by :func:`load`.
    output
        If given, write the figure to this path (png/pdf/svg). Otherwise only
        return the Figure object.
    show
        If True, call ``plt.show()`` at the end.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Keep the matplotlib import inside the function so the module itself can
    # be imported in headless contexts without pulling GUI deps.
    import matplotlib.pyplot as plt

    alloc_cols = [c for c in df.columns if c.startswith("arena_") and c.endswith("_allocated_mb")]
    df = df.copy()
    df["arena_total_allocated_mb"] = df[alloc_cols].sum(axis=1, min_count=1)

    fig, (ax_arena, ax_proc) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Top: AMReX arenas, one line per rank.
    for rank, g in df.groupby("rank"):
        ax_arena.plot(g["step"], g["arena_total_allocated_mb"], label=f"rank {rank}")
    ax_arena.set_ylabel("AMReX arenas total\nallocated (MB)")
    ax_arena.grid(True, alpha=0.3)
    if df["rank"].nunique() <= 16:
        ax_arena.legend(loc="best", fontsize=8, ncol=2)

    # Bottom: process memory (RSS solid, HWM dashed). Values already in MB.
    has_rss = "vm_rss_mb" in df.columns and df["vm_rss_mb"].notna().any()
    has_hwm = "vm_hwm_mb" in df.columns and df["vm_hwm_mb"].notna().any()

    if has_rss or has_hwm:
        for rank, g in df.groupby("rank"):
            color = None
            if has_rss:
                (line,) = ax_proc.plot(
                    g["step"], g["vm_rss_mb"], label=f"rank {rank} VmRSS"
                )
                color = line.get_color()
            if has_hwm:
                ax_proc.plot(
                    g["step"],
                    g["vm_hwm_mb"],
                    linestyle="--",
                    color=color,
                    alpha=0.7,
                )
        ax_proc.set_ylabel("Process memory (MB)\nsolid = VmRSS, dashed = VmHWM")
    else:
        ax_proc.text(
            0.5,
            0.5,
            "No /proc/self/status data (non-Linux run?)",
            ha="center",
            va="center",
            transform=ax_proc.transAxes,
        )

    ax_proc.set_xlabel("step")
    ax_proc.grid(True, alpha=0.3)

    fig.suptitle("WarpX MemoryPerRank diagnostic", y=0.995)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Read WarpX MemoryPerRank YAML output and plot per-rank memory over time."
    )
    parser.add_argument(
        "directory",
        help="Path to the MemoryPerRank output directory "
        "(typically diags/reducedfiles/<reduced_diags_name>/).",
    )
    parser.add_argument(
        "--prefix",
        default="MPR",
        help="File prefix used by the diagnostic (default: MPR).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the figure to this path instead of only returning it.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Also export the flattened DataFrame to this CSV path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the plot in an interactive window.",
    )
    args = parser.parse_args(argv)

    df = load(args.directory, prefix=args.prefix)
    if args.csv:
        df.to_csv(args.csv, index=False)
    plot(df, output=args.output, show=args.show)
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
