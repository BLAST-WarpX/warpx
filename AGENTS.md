# AGENTS Guidelines for This Repository

## Project Overview

WarpX is a massively parallel electromagnetic Particle-In-Cell (PIC) code built on top of AMReX (adaptive mesh refinement framework). It supports multiple dimensionalities (1D, 2D, 3D, RZ cylindrical) and compute backends (CPU/OpenMP, CUDA, HIP, SYCL).

## Development Environment

If you cannot find the `cmake` or `ctest` command, activate the conda environment named `warpx-cpu-mpich-dev` before running shell commands that compile or test WarpX:
```bash
conda activate warpx-cpu-mpich-dev
```
If this environment does not yet exist, create it as described in `Docs/source/install/cmake.rst`.

## Build Commands

The cmake build directory is always inside the repository root (or worktree root). Never look for or create a build directory outside of the current working directory.

```bash
# Configure (common development build with Python bindings)
cmake --fresh -S . -B build -DWarpX_DIMS=3 -DWarpX_PYTHON=ON

# Build
cmake --build build -j 8

# Install Python bindings
cmake --build build --target pip_install

# Faster rebuild (skip dependency checks)
cmake --build build -j 8 --target pip_install_nodeps

# Faster link times during development
cmake -S . -B build -DWarpX_PYTHON=ON -DWarpX_PYTHON_IPO=OFF -DpyAMReX_IPO=OFF
```

Key CMake options:
- `-DWarpX_DIMS=3` — dimensionality: `1`, `2`, `3`, `RZ` (or semicolon-separated for multiple)
- `-DWarpX_COMPUTE=OMP` — backend: `NOACC`, `OMP`, `CUDA`, `SYCL`, `HIP`
- `-DWarpX_PYTHON=ON` — Python bindings
- `-DWarpX_OPENPMD=ON` — HDF5/ADIOS I/O
- `-DWarpX_FFT=ON` — spectral solvers (PSATD)
- `-DWarpX_QED=ON` — QED physics

## Testing

Tests use CTest. Each test has an input file, an analysis script, and a checksum file.

```bash
# List tests
ctest --test-dir build -N

# Run all tests
ctest --test-dir build --output-on-failure

# Run specific test by regex
ctest --test-dir build -R laser_acceleration

# Run exact test
ctest --test-dir build -R "test_3d_langmuir_multi\..*"

# Exclude slow tests
ctest --test-dir build -LE slow
```

Test output goes to `build/bin/<test_name>/`.

- When running tests: ignore checksum failures, since they can be platform-dependent.
- When debugging/fixing tests: do not modify the tolerance of assert statements in the Python analysis files just to make the tests pass (unless explicitly asked to do so).

### Adding a Test

Use `add_warpx_test()` in the test directory's `CMakeLists.txt`. Generate checksums with `CHECKSUM_RESET=ON ctest --test-dir build -R your_test_name`.

Tests must be quick to run on a 2 core CI CPU runner (ideally <30sec) and be written in a CPU/GPU portable way.
Tests have analysis functions to validate their outputs are as expected.

## Linting and Formatting

Pre-commit hooks handle formatting:
```bash
python -m pip install -U pre-commit
pre-commit install
pre-commit run -a  # run on all files
```

- **C++**: clang-format (4 spaces, 100 char line limit)
- **Python**: Ruff (configured in `pyproject.toml`)

Commits should limit any formatting changes of unchanged code.

## Code Architecture

### Source Layout

- `Source/WarpX.H/.cpp` — main simulation class (inherits `amrex::AmrCore`)
- `Source/Evolve/` — time evolution loop (`WarpX::Evolve()`, `OneStep_nosub()`)
- `Source/FieldSolver/` — Maxwell/Poisson solvers (finite-difference, spectral/PSATD, electrostatic, magnetostatic)
- `Source/Particles/` — particle containers, pushers (Boris, Vay), deposition, gathering, collisions
- `Source/Diagnostics/` — output and diagnostics
- `Source/BoundaryConditions/` — PML, PEC
- `Source/Initialization/` — initial conditions, external fields
- `Source/Laser/` — laser injection
- `Source/ablastr/` — abstraction layer shared with other codes
- `Source/Python/` — pybind11 bindings
- `Python/pywarpx/` — Python/PICMI interface
- `Examples/` — test cases and physics applications

### Key Patterns

- Each dimensionality builds a separate executable/library with a dimension suffix
- Fields are stored as AMReX `MultiFab` objects, managed via `ablastr::fields::MultiFabRegister`
- Particle species managed by `MultiParticleContainer` → `WarpXParticleContainer`
- Compile-time macros: `WARPX_DIM_3D`, `WARPX_DIM_XZ`, `WARPX_DIM_1D_Z`, `WARPX_DIM_RZ`

## C++ Style

- 4 spaces indentation, no tabs, max 100 chars/line
- Member variables prefixed with `m_`
- CamelCase for files and classes
- Space before `()` in function definitions, not in calls
- Opening brace `{` on new line for functions/classes
- Always use curly braces for single-statement blocks
- Use `amrex::` namespace prefix (avoid `using namespace amrex`)

### Include Order

In `.cpp` files: (1) corresponding header, (2) WarpX headers, (3) WarpX forward declarations, (4) AMReX headers, (5) AMReX forward declarations, (6) third-party headers, (7) standard library. Each group alphabetically sorted with blank lines between groups.

## Backward Compatibility

When a change removes or renames a user-facing input parameter, add a guard to the relevant `BackwardCompatibility()` method — e.g., in `Source/WarpX.cpp` or `Source/Particles/PhysicalParticleContainer.cpp`.

## Version Control

- Main branch: `development` (not `main`)
- Fork-and-branch workflow; PRs target `development`
- Pull requests with features and bug fixes need to add a test for coverage.
