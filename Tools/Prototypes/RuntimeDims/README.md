# Runtime-Dimensionality Prototype

This prototype validates the planned WarpX architecture in which

- AMReX is built **once**, with `AMREX_SPACEDIM=3`, instead of once per
  dimensionality, and
- a **single binary** supports all dimensionalities, selected at runtime via
  the existing `geometry.dims` input parameter, and
- per-dimension code paths are expressed as `template <warpx::Dim D>` kernels
  with `if constexpr` branches (replacing the `#if defined(WARPX_DIM_*)`
  macro snippets 1:1), instantiated for all dimensionalities and dispatched
  at runtime (`warpx::dim_dispatch`), mirroring how WarpX already dispatches
  the runtime particle shape order to compile-time template instantiations.

## Layout conventions (unified, degenerate-3D)

- Fields of all dimensionalities are stored in 3D `MultiFab`s with collapsed
  dimensions of extent 1: 1D uses `(1, 1, nz)`, 2D uses `(nx, 1, nz)`; z is
  always at index 2 (`warpx::IdxMap`, replacing `WARPX_ZINDEX`).
- Collapsed dimensions use `prob_lo/hi = [-0.5, 0.5)` (cell size exactly 1 m,
  matching the existing `WarpX::CellSize` convention for absent dimensions),
  zero guard cells, cell-centered staggering, and periodic boundaries.
- Particle positions are always three SoA components (`x`, `y`, `z`);
  components of collapsed dimensions are 0 and never updated by the pusher
  (e.g., 2D tracks `uy` but not `y`, exactly like the native 2D code).
- Shape factors are not evaluated along collapsed dimensions; the deposition
  and gather loop nests degenerate at compile time to exactly the loop nests
  of the native per-dimension code.

## Building

```bash
cmake -S . -B build -DWarpX_DIMS="1;2;3" -DWarpX_RUNTIME_DIMS_PROTO=ON
cmake --build build -j 6 --target app_unified app_1d app_2d app_3d
```

This builds `build/bin/warpx.unified` (linked only against `amrex_3d` and
compiled with `WARPX_DIM_RUNTIME`; it does not link the per-dimension WarpX
libraries) plus the native binaries used for comparison. With
`-DWarpX_DIMS=3`, AMReX is built exactly once — the intended end state.

## Running

The prototype consumes (a subset of) the standard WarpX inputs format:

```bash
cd build/bin
./warpx.unified ../../Examples/Tests/langmuir/inputs_test_1d_langmuir_multi algo.current_deposition=direct
./warpx.unified ../../Examples/Tests/langmuir/inputs_test_2d_langmuir_multi algo.current_deposition=direct
./warpx.unified ../../Examples/Tests/langmuir/inputs_test_3d_langmuir_multi algo.current_deposition=direct
```

Supported physics (the prototype slice): explicit Yee FDTD, Boris pusher,
direct current deposition (shape orders 1-4, energy-conserving gather),
periodic boundaries, `NUniformPerCell` injection with constant density and
parsed momentum functions, serial/MPI/OpenMP on CPU.

## Validation

- `analysis/analysis_langmuir.py <plotfile>` checks the fields against the
  analytic Langmuir-wave solution with the tolerances of the corresponding
  native tests.
- `analysis/compare_native.py <unified_plt> <native_plt> [tol]` compares
  against a native binary run on the same inputs; both codes cell-center
  diagnostics with the exact same averaging. Measured agreement of serial
  runs (`mpirun -np 1`, `OMP_NUM_THREADS=1`, 40 steps): particle positions
  and weights are **bitwise identical**; fields agree to a few 1e-14
  relative (single-box runs: half the components bitwise, the rest within
  1-2 ULP). The residue stems from per-cell floating-point summation-order
  effects in cells with multiple deposition contributors; the test
  tolerance is 1e-12. B components are normalized against
  `max(|B|, |E|/c)`, since B is numerical noise in the electrostatic-like
  Langmuir problem.

Native comparison runs use the same inputs plus
`algo.current_deposition=direct`.

## Performance

`perf/run_perf.sh <build_dir> [steps] [threads]` runs `warpx.unified` and the
native binaries on identical Langmuir problems and reports the TinyProfiler
hot regions.

Measured exclusive times (serial CPU, Release, GCC 13, 100 steps,
`OMP_NUM_THREADS=1`; Langmuir-multi with direct deposition, shape order 1):

| Case (cells, ppc/species)   | Region            | unified [s] | native [s] | delta |
|-----------------------------|-------------------|------------:|-----------:|------:|
| 1D (2^20, 8); 16.8M prt     | gather + push     |        80.9 |       76.5 |  +6 % |
|                             | current deposition|        38.6 |       29.2 | +32 % |
| 2D (1024^2, 4x4); 33.5M prt | gather + push     |       224.6 |      206.0 |  +9 % |
|                             | current deposition|        79.6 |       72.2 | +10 % |
| 3D (128^3, 2x2x2); 33.5M prt| gather + push     |       339.7 |      287.1 | +18 % |
|                             | current deposition|       109.3 |      112.2 |  -3 % |

Interpretation:

- The 3D case is the *control*: its kernels instantiate to the same code as
  the native 3D binary, so its +18 % gather delta measures the overhead of
  the prototype *harness*, not of the dimension templating. The driver
  iterates particles per box, while native WarpX iterates per tile with
  cache-resident field data; the same locality gap explains the 1D
  deposition delta (the driver deposits into a full-box local array, native
  into small tile-local arrays).
- The 1D and 2D deltas are *smaller than or equal to* the 3D control in the
  gather, i.e., the runtime-dimensionality mechanics (degenerate-3D
  indexing, `if constexpr` kernels, `dim_dispatch`) add no measurable
  dimension-specific cost on top of the harness effects.
- Structural particle-memory overhead of the unified layout (positions
  always 3 SoA reals): +40 % in 1D (7 vs 5 reals), +17 % in 2D (7 vs 6),
  0 % in 3D.
- The native binaries also run particle boundary checks
  (`ApplyBoundaryConditions`) that the prototype folds into `Redistribute`,
  so total wall times are not directly comparable; the per-region kernel
  times above are.

Adopting WarpX's tiled particle iteration in the full migration (phases A-D)
removes the harness-locality gap by construction, since the converted
kernels are called from the existing containers.

## Known limitations / next steps

- RZ (and RCYLINDER/RSPHERE) are designed for (the `warpx::Dim` enum, traits
  and `PIdx` cover them) but not yet dispatched at runtime; RZ additionally
  needs AMReX enablers for the MLMG `setRZ`/`setRZCorrection` paths, which
  are compiled out for `AMREX_SPACEDIM != 2` builds.
- Unconverted kernels (Esirkepov/Villasenor/Vay deposition, implicit
  pushers/gathers, shared-memory deposition) are guarded with
  `#if !defined(WARPX_DIM_RUNTIME)` and keep compiling unchanged in
  per-dimension builds; they are converted in later migration phases.
- Particle initialization in the driver stages on the host (CPU); GPU runs
  of the prototype require building the staging vectors as pinned memory
  (the kernels themselves are CPU/GPU portable).
