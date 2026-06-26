.. _memoryperrank-plotting:

Reading and plotting ``MemoryPerRank`` output
==============================================

WarpX provides via :ref:`reduced diagnostics <dataanalysis-formats-reduced>` an
output :ref:`MemoryPerRank <running-cpp-parameters-diagnostics>`, which
captures detailed per-rank memory usage over time (AMReX arenas, host-process
memory, hostname, and GPU device id).

Each participating MPI rank writes one YAML file
``<path>/<file_prefix>.<zero-padded rank>.yaml``. Every diagnostic interval
appends a new self-contained YAML document (separated by ``---``), so the
output can be streamed cheaply with ``yaml.safe_load_all()``.


Generating the data
-------------------

To enable the diagnostic, add the following lines to your input file
(``MPR`` is an arbitrary name, ``100`` is the step interval):

.. code-block:: text

    warpx.reduced_diags_names = MPR
    MPR.type        = MemoryPerRank
    MPR.intervals   = 100
    MPR.path        = "diags/reducedfiles/MemoryPerRank/"
    MPR.file_prefix = "MPR"
    # Optional: only every 4th rank writes a file (useful on large runs).
    # MPR.rank_stride = 4

Or from a PICMI script:

.. code-block:: python

    mpr = picmi.ReducedDiagnostic(
        diag_type="MemoryPerRank",
        name="MPR",
        period=100,
        path="diags/reducedfiles/MemoryPerRank/",
        file_prefix="MPR",
    )
    sim.add_diagnostic(mpr)


Loading the data
----------------

A small Python helper in
:download:`read_memory_per_rank.py <../../../../Tools/PostProcessing/read_memory_per_rank.py>`
collects all per-rank YAML files in a directory into a tidy
``pandas.DataFrame``. Each row corresponds to one ``(rank, step)`` snapshot:

.. code-block:: python

    import read_memory_per_rank as mpr

    df = mpr.load("diags/reducedfiles/MemoryPerRank/", prefix="MPR")
    print(df.columns.tolist())
    # ['file', 'step', 'time', 'rank', 'nprocs', 'hostname',
    #  'gpu_device_id', 'gpu_total_mb', 'gpu_free_mb',
    #  'arena_main_allocated_mb', 'arena_main_used_mb',
    #  'arena_managed_allocated_mb', 'arena_managed_used_mb',
    #  'arena_pinned_allocated_mb', 'arena_pinned_used_mb',
    #  'arena_comms_allocated_mb', 'arena_comms_used_mb',
    #  'vm_peak_mb', 'vm_size_mb', 'vm_hwm_mb', 'vm_rss_mb']

    # All memory columns are in MB (float, 3-decimal precision).
    # Max resident-set size over all ranks at each step:
    df.groupby("step")[["vm_rss_mb", "vm_hwm_mb"]].max()


Plotting per-rank memory over time
----------------------------------

The same helper exposes a one-line plotting function:

.. code-block:: python

    import matplotlib.pyplot as plt
    import read_memory_per_rank as mpr

    df = mpr.load("diags/reducedfiles/MemoryPerRank/", prefix="MPR")
    fig = mpr.plot(df, output="mpr_timeline.png")
    plt.show()

The figure adapts to the run: on GPU builds the headline panel shows GPU
memory used per rank (the quantity that drives device OOM); on CPU-only
builds it shows total AMReX arena allocation. A host-process ``VmRSS`` /
``VmHWM`` panel is always included.

The per-rank rendering mode is auto-selected from the rank count (one
curve per rank for small runs, an envelope of min/median/max for medium
runs, a rank x step heatmap for very large runs). See the script's
module docstring and ``--help`` for the available modes and overrides.

You can also run the helper as a script:

.. code-block:: bash

    python3 Tools/PostProcessing/read_memory_per_rank.py \\
        diags/reducedfiles/MemoryPerRank/ \\
        --prefix MPR \\
        --output mpr_timeline.png \\
        --csv    mpr_timeline.csv \\
        --mode   auto


Spotting imbalance between ranks
--------------------------------

For a quick per-rank spread at any given step:

.. code-block:: python

    at_step = df[df["step"] == df["step"].max()]
    print(at_step[["rank", "hostname", "gpu_device_id",
                   "arena_main_allocated_mb", "vm_rss_mb"]]
          .sort_values("vm_rss_mb", ascending=False)
          .head(10))

A rank whose ``arena_main_allocated_mb`` is an outlier is carrying a larger
domain chunk; a rank whose ``vm_rss_mb`` is an outlier but whose arena size is
not typically indicates memory consumed outside AMReX (MPI staging, Python, ...).
Correlating with the :ref:`LoadBalanceCosts <running-cpp-parameters-diagnostics>`
diagnostic is often useful when both are enabled.
