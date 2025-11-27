.. _install-dependencies:

List of Dependencies
====================

WarpX depends on the following popular third party software.
Please see installation instructions below.

- a mature `C++17 <https://en.wikipedia.org/wiki/C%2B%2B17>`__ compiler, e.g., GCC 9.1+, Clang 7, NVCC 11.0, MSVC 19.15 or newer
- `CMake 3.24.0+ <https://cmake.org>`__
- `Git 2.18+ <https://git-scm.com>`__
- `AMReX <https://amrex-codes.github.io>`__: we automatically download and compile a copy of AMReX
- `PICSAR <https://github.com/ECP-WarpX/picsar>`__: we automatically download and compile a copy of PICSAR

and for Python bindings:

- `pyAMReX <https://github.com/AMReX-Codes/pyamrex>`__: we automatically download and compile a copy of pyAMReX
- `pybind11 <https://github.com/pybind/pybind11>`__: we automatically download and compile a copy of pybind11

Optional dependencies include:

- `MPI 3.0+ <https://www.mpi-forum.org/docs/>`__: for multi-node and/or multi-GPU execution
- for on-node accelerated compute *one of either*:

  - `OpenMP 3.1+ <https://www.openmp.org>`__: for threaded CPU execution or
  - `CUDA Toolkit 11.7+ <https://developer.nvidia.com/cuda-downloads>`__: for Nvidia GPU support (see `matching host-compilers <https://gist.github.com/ax3l/9489132>`__) or
  - `ROCm 5.2+ (5.5+ recommended) <https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-rocm-installation-readme/>`__: for AMD GPU support
  - `oneAPI <https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html>`__: for Intel GPU support
- `FFTW3 <http://www.fftw.org>`__: for spectral solver (PSATD or IGF) support when running on CPU or SYCL

  - also needs the ``pkg-config`` tool on Unix
- `BLAS++ <https://github.com/icl-utk-edu/blaspp>`__ and `LAPACK++ <https://github.com/icl-utk-edu/lapackpp>`__: for spectral solver (PSATD) support in RZ geometry
- `Boost 1.66.0+ <https://www.boost.org/>`__: for QED lookup tables generation support
- `openPMD-api 0.16.1+ <https://github.com/openPMD/openPMD-api>`__: we automatically download and compile a copy of openPMD-api for openPMD I/O support

  - see `optional I/O backends <https://github.com/openPMD/openPMD-api#dependencies>`__, i.e., ADIOS2 and/or HDF5
- `Ascent 0.8.0+ <https://ascent.readthedocs.io>`__: for in situ 3D visualization
- `SENSEI 4.0.0+ <https://sensei-insitu.org>`__: for in situ analysis and visualization
- `CCache <https://ccache.dev>`__: to speed up rebuilds (For CUDA support, needs version 3.7.9+ and 4.2+ is recommended)
- `Ninja <https://ninja-build.org>`__: for faster parallel compiles
- `Python 3.9+ <https://www.python.org>`__

  - `mpi4py <https://mpi4py.readthedocs.io>`__
  - `numpy <https://numpy.org>`__
  - `periodictable <https://periodictable.readthedocs.io>`__
  - `picmistandard <https://picmi-standard.github.io>`__
  - `lasy <https://lasydoc.readthedocs.io>`__
  - see our ``requirements.txt`` file for compatible versions

If you are on a high-performance computing (HPC) system, then :ref:`please see our separate HPC documentation <install-hpc>`.

For all other systems, we recommend to use a **package dependency manager**:
Pick *one* of the installation methods below to install all dependencies for WarpX development in a consistent manner.
