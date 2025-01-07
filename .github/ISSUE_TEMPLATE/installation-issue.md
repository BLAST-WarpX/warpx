---
name: Installation issue
about: Report an issue with installing or setting up WarpX.
labels: [install]
---

_Please remove any sensitive information (e.g., passwords, API keys) from your submission.
Please check the relevant boxes and fill in the specific versions or details for the relevant items.
Thank you for taking the time to report this issue. We will respond as soon as possible._

## Description
A clear and concise description of the issue.

## System information
- Operating system (name and version):
  - [ ] Linux: e.g., Ubuntu 22.04 LTS
  - [ ] macOS: e.g., macOS Monterey 12.4
  - [ ] Windows: e.g., Windows 11 Pro
- Version of WarpX: e.g., latest, 24.10, etc.
- Installation method:
  - [ ] Conda
  - [ ] Spack
  - [ ] PyPI
  - [ ] Brew
  - [ ] From source with CMake
  - [ ] Module system on an HPC cluster
- Other dependencies: yes/no, describe
- Computational resources:
  - [ ] CPU
  - [ ] GPU: e.g., NVIDIA, AMD, etc.

If you encountered the issue on an HPC cluster, please check our [HPC documentation](https://warpx.readthedocs.io/en/latest/install/hpc.html) to see if your HPC cluster is already supported.

If you encountered the issue installing from source with CMake, please provide the output of the following steps:
1. buildsystem generation: output of `cmake --fresh -S . -B build` (include your specific build options, e.g., `-DWarpX_DIMS=3`)
2. project build: output of `cmake --build build` (include your specific build options, e.g., `-j 4`)

If applicable, please add any additional information about your software environment:
- [ ] CMake: e.g., 3.24.0
- [ ] C++ compiler: e.g., GNU 11.3 with NVCC 12.0.76
- [ ] Python: e.g., CPython 3.12
- [ ] MPI: e.g., OpenMPI 4.1.1
- [ ] FFTW: e.g., 3.3.10
- [ ] HDF5: e.g., 1.14.0
- [ ] ADIOS2: e.g., 2.10.0
- Other dependencies: yes/no, describe

## Additional information
If applicable, please add any additional information that may help explain the issue, such as log files (e.g., build logs, error logs, etc.), error messages (e.g., compiler errors, runtime errors, etc.), screenshots, or other relevant details.
