#!/usr/bin/env python3
"""
Read and visualize WarpX ``MemoryPerRank`` reduced-diagnostic output.

Each participating MPI rank writes one file named
``<path>/<file_prefix>.<zero-padded rank>.yaml`` containing a stream of YAML
documents (one per diagnostic interval). This helper loads all rank files
from a directory into a tidy ``pandas.DataFrame`` and optionally produces
summary plots tailored to the run type:

* On GPU builds the headline panel is **GPU memory used per rank**
  (``gpu_total_mb - gpu_free_mb``), the quantity that actually drives OOM
  on production GPU clusters. A second panel breaks down the device-resident
  AMReX arenas (``main`` + ``device`` + ``managed``), so a divergence
  between the two panels means memory pressure outside AMReX (CUDA runtime,
  MPI staging, plugin libraries, ...).
* On CPU-only builds the headline panel is the total AMReX arena
  allocation per rank.
* Both build types also get a host-process panel (``VmRSS`` / ``VmHWM``).

The per-rank rendering mode is auto-detected from the rank count and can
be overridden via the ``mode`` parameter / ``--mode`` flag:

* ``spaghetti`` -- one labelled curve per rank (good for <= 32 ranks).
* ``band`` -- faint per-rank background, with bold min/median/max
  envelopes overlaid (good for ~32-256 ranks).
* ``heatmap`` -- rank on the Y axis, step on the X axis, value encoded as
  color (good for >256 ranks; one heatmap per panel).

Requires: pyyaml, pandas, matplotlib.

Usage (as a library)::

    import read_memory_per_rank as mpr
    df = mpr.load("diags/reducedfiles/MemoryPerRank/", prefix="MPR")
    fig = mpr.plot(df)              # mode="auto"
    fig = mpr.plot(df, mode="band") # force a specific mode

Usage (as a script)::

    python3 read_memory_per_rank.py diags/reducedfiles/MemoryPerRank/ \\
        --prefix MPR --output mpr_timeline.png --mode auto
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


# Arenas that are device-resident on a GPU build (i.e. their allocations
# count toward GPU memory pressure). `pinned` and `comms` are host-side
# staging buffers and intentionally excluded from this list.
_GPU_SIDE_ARENAS = (
    "arena_main_allocated_mb",
    "arena_device_allocated_mb",
    "arena_managed_allocated_mb",
)


def _pick_mode(n_ranks: int) -> str:
    """Auto-pick a rendering mode based on rank count."""
    if n_ranks <= 32:
        return "spaghetti"
    if n_ranks <= 256:
        return "band"
    return "heatmap"


def _draw_panel(ax, df: pd.DataFrame, x: str, y: str, ylabel: str,
                mode: str, n_ranks: int):
    """Render a single per-rank time-series panel in the requested mode.

    The panel renders a quantity ``y`` over an axis ``x`` (typically
    ``"step"``). The Y axis (or colorbar in heatmap mode) is labelled
    with ``ylabel``.
    """
    import matplotlib.pyplot as plt  # local: keeps module import headless

    if mode == "spaghetti":
        for rank, g in df.groupby("rank"):
            ax.plot(g[x], g[y], label=f"rank {rank}", linewidth=1)
        if n_ranks <= 16:
            ax.legend(loc="best", fontsize=8, ncol=2)
        ax.set_ylabel(ylabel)
    elif mode == "band":
        # Faint per-rank background lines.
        for _, g in df.groupby("rank"):
            ax.plot(g[x], g[y], color="grey", alpha=0.15, linewidth=0.7)
        # Bold min/median/max envelopes across ranks at each x.
        agg = df.groupby(x)[y].agg(["min", "median", "max"]).dropna()
        if not agg.empty:
            ax.plot(agg.index, agg["max"], color="red",
                    linewidth=2, label="max")
            ax.plot(agg.index, agg["median"], color="black",
                    linewidth=1.5, label="median")
            ax.plot(agg.index, agg["min"], color="blue",
                    linewidth=1, alpha=0.7, label="min")
            ax.legend(loc="best", fontsize=8)
        ax.set_ylabel(ylabel)
    elif mode == "heatmap":
        pivot = df.pivot_table(index="rank", columns=x, values=y, aggfunc="last")
        if pivot.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_ylabel("rank")
            return
        im = ax.imshow(
            pivot.values, aspect="auto", origin="lower",
            extent=[float(pivot.columns.min()), float(pivot.columns.max()),
                    float(pivot.index.min()), float(pivot.index.max())],
        )
        plt.colorbar(im, ax=ax, label=ylabel)
        ax.set_ylabel("rank")
    else:
        raise ValueError(
            f"Unknown mode {mode!r}; expected one of: auto, spaghetti, band, heatmap."
        )


def plot(
    df: pd.DataFrame,
    output: str | None = None,
    show: bool = False,
    mode: str = "auto",
):
    """Plot ``MemoryPerRank`` time series, with panels picked from the data.

    On a **GPU build** (when the YAML carries a ``gpu:`` block):

    * Top: per-rank GPU memory used (``gpu_total_mb - gpu_free_mb``) with
      a dashed line at the (median) device total. This is the panel that
      actually answers "how close are we to OOM?".
    * Middle: per-rank device-resident AMReX arenas, summed
      (``main + device + managed``). Diverging from the top panel means
      memory pressure that AMReX does not see (CUDA runtime, MPI staging,
      plugin libraries, ...).
    * Bottom: per-rank host-process memory (``VmRSS`` / ``VmHWM``).

    On a **CPU-only build** the GPU panels are skipped and the figure has
    only the total-arena and host-process panels.

    Parameters
    ----------
    df
        DataFrame as returned by :func:`load`.
    output
        If given, write the figure to this path (png/pdf/svg). Otherwise
        only return the Figure object.
    show
        If True, call ``plt.show()`` at the end.
    mode
        Rendering mode for per-rank curves:
        ``"auto"`` (default) picks ``spaghetti``, ``band``, or ``heatmap``
        based on rank count; or pass one of those names directly.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Keep the matplotlib import inside the function so the module itself can
    # be imported in headless contexts without pulling GUI deps.
    import matplotlib.pyplot as plt

    df = df.copy()
    n_ranks = df["rank"].nunique()
    if mode == "auto":
        mode = _pick_mode(n_ranks)

    has_gpu = (
        "gpu_total_mb" in df.columns and df["gpu_total_mb"].notna().any()
    )

    # Build the list of panels to draw, top to bottom.
    panels: list[dict] = []

    if has_gpu:
        df["gpu_used_mb"] = df["gpu_total_mb"] - df["gpu_free_mb"]
        panels.append({
            "y": "gpu_used_mb",
            "ylabel": "GPU memory used (MB)\n(total \u2212 free)",
            "ceiling_col": "gpu_total_mb",
        })
        device_cols = [c for c in _GPU_SIDE_ARENAS if c in df.columns]
        if device_cols:
            df["gpu_arena_total_mb"] = df[device_cols].sum(axis=1, min_count=1)
            panels.append({
                "y": "gpu_arena_total_mb",
                "ylabel": "Device-side AMReX arenas\nallocated (MB)",
            })
    else:
        alloc_cols = [
            c for c in df.columns
            if c.startswith("arena_") and c.endswith("_allocated_mb")
        ]
        df["arena_total_allocated_mb"] = df[alloc_cols].sum(axis=1, min_count=1)
        panels.append({
            "y": "arena_total_allocated_mb",
            "ylabel": "AMReX arenas total\nallocated (MB)",
        })

    # Host-process panel: always last when present.
    has_rss = "vm_rss_mb" in df.columns and df["vm_rss_mb"].notna().any()
    has_hwm = "vm_hwm_mb" in df.columns and df["vm_hwm_mb"].notna().any()
    if has_rss:
        panels.append({
            "y": "vm_rss_mb",
            "ylabel": (
                "Process memory (MB)\nsolid = VmRSS, dashed = VmHWM"
                if has_hwm and mode == "spaghetti"
                else "Process VmRSS (MB)"
            ),
            "extra_y": "vm_hwm_mb" if (has_hwm and mode == "spaghetti") else None,
        })

    fig, axes = plt.subplots(
        len(panels), 1,
        figsize=(10, 3 + 2.5 * len(panels)),
        sharex=(mode != "heatmap"),
        squeeze=False,
    )
    axes = axes[:, 0]

    for ax, panel in zip(axes, panels):
        _draw_panel(ax, df, "step", panel["y"], panel["ylabel"],
                    mode, n_ranks)
        ax.grid(True, alpha=0.3)

        # Optional ceiling line (e.g. GPU total memory). Only meaningful in
        # spaghetti / band modes; in heatmap the colorbar already shows scale.
        ceiling_col = panel.get("ceiling_col")
        if ceiling_col and ceiling_col in df.columns and mode != "heatmap":
            ceiling = df[ceiling_col].median()
            ax.axhline(ceiling, color="grey", linestyle="--", linewidth=1,
                       label=f"GPU total ({ceiling:.0f} MB)")
            # Re-draw legend so the ceiling line shows up alongside any
            # per-rank / band labels.
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="best", fontsize=8)

        # Spaghetti VmHWM overlay (dashed).
        extra_y = panel.get("extra_y")
        if extra_y and extra_y in df.columns and mode == "spaghetti":
            for _, g in df.groupby("rank"):
                ax.plot(g["step"], g[extra_y], linestyle="--",
                        alpha=0.5, linewidth=0.8)

    # In heatmap mode the X axis on every panel shows step; in others only
    # the bottom panel labels it (sharex collapses the rest).
    if mode == "heatmap":
        for ax in axes:
            ax.set_xlabel("step")
    else:
        axes[-1].set_xlabel("step")

    fig.suptitle(
        f"WarpX MemoryPerRank \u2014 {n_ranks} ranks, mode={mode}",
        y=0.995,
    )
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
    parser.add_argument(
        "--mode",
        choices=("auto", "spaghetti", "band", "heatmap"),
        default="auto",
        help=(
            "Per-rank rendering mode. 'auto' picks: spaghetti for <=32 ranks, "
            "band for <=256 ranks, heatmap above. 'spaghetti' = one labelled "
            "curve per rank. 'band' = faint per-rank lines plus bold "
            "min/median/max envelopes. 'heatmap' = rank x step image with "
            "value encoded as color (one heatmap per panel)."
        ),
    )
    args = parser.parse_args(argv)

    df = load(args.directory, prefix=args.prefix)
    if args.csv:
        df.to_csv(args.csv, index=False)
    plot(df, output=args.output, show=args.show, mode=args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
