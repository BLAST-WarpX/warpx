---
name: Installation issue
about: Report an issue with installing or setting up WarpX
labels: [install]
---

_Please remove any sensitive information (e.g., passwords, API keys) from your submission. Thank you for taking the time to report this issue. We will respond as soon as possible._

**Where are you trying to install WarpX?**
A clear and concise description of where you want to install WarpX.

**Tell us about your software environment**
- operating system: [name and version]
- version of WarpX: [e.g., latest, 24.09, etc.]
- HPC machine: [Are you running on an HPC cluster? We might support it already: https://warpx.readthedocs.io/en/latest/install/hpc.html]

**What have you tried so far?**
- installing WarpX via: [e.g., conda-forge, spack, pip, brew, from source, module system, etc.]

**Are you trying to install from source?**
Please provide:
- the output of `cmake --fresh -S . -B build`
- the output of `cmake --build build`

Let us also know more about your software environment (if applicable):
- version of CMake: [e.g., 3.24.0]
- name and version of C++ compiler: [e.g., GNU 11.3 with NVCC 12.0.76]
- name and version of Python implementation: [e.g., CPython 3.12]
- name and version of MPI: [e.g., OpenMPI 4.1.1]
- version of FFTW: [e.g., 3.3.10]
- version of HDF5: [e.g., 1.14.0]
- version of ADIOS2: [e.g., 2.10.0]
- other relevant dependency names and versions: [e.g., NumPy 2.0, Ascent 0.8.0, etc.]

**Additional information**
If applicable, please add other info to help explain the issue.

**Screenshots**
If applicable, please add screenshots to help explain the issue.
