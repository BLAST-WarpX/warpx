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
  against a native binary run on the same inputs. Both codes cell-center
  diagnostics with the exact same averaging, so serial runs
  (`mpirun -np 1`, `OMP_NUM_THREADS=1`) must agree **bitwise** (tol 0);
  parallel runs agree up to atomic-reduction rounding (tol ~1e-9).

Native comparison runs use the same inputs plus
`algo.current_deposition=direct`.

## Performance

(Results table to be filled by the performance harness; see the plan in the
PR description: per-TinyProfiler-region times of unified vs. native runs for
1D/2D/3D Langmuir problems at various sizes, plus Arena peak-memory reports.)

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
