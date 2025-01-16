.. _install-dependencies:

Dependencies
============

WarpX depends on the following popular third party software.
Please see installation instructions below.

- a mature `C++17 <https://en.wikipedia.org/wiki/C%2B%2B17>`__ compiler, e.g., GCC 8.4+, Clang 7, NVCC 11.0, MSVC 19.15 or newer
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
- `FFTW3 <http://www.fftw.org>`__: for spectral solver (PSATD or IGF) support when running on CPU or SYCL

  - also needs the ``pkg-config`` tool on Unix
- `BLAS++ <https://github.com/icl-utk-edu/blaspp>`__ and `LAPACK++ <https://github.com/icl-utk-edu/lapackpp>`__: for spectral solver (PSATD) support in RZ geometry
- `Boost 1.66.0+ <https://www.boost.org/>`__: for QED lookup tables generation support
- `openPMD-api 0.15.1+ <https://github.com/openPMD/openPMD-api>`__: we automatically download and compile a copy of openPMD-api for openPMD I/O support

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


Conda (Linux/macOS/Windows)
---------------------------

`Conda <https://conda.io>`__/`Mamba <https://mamba.readthedocs.io>`__ are cross-compatible, user-level package managers.

.. tip::

   We recommend to configure your conda to use the faster ``libmamba`` `dependency solver <https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community>`__.

   .. code-block:: bash

      conda update -y -n base conda
      conda install -y -n base conda-libmamba-solver
      conda config --set solver libmamba

   We recommend to deactivate that conda self-activates its ``base`` environment.
   This `avoids interference with the system and other package managers <https://collegeville.github.io/CW20/WorkshopResources/WhitePapers/huebl-working-with-multiple-pkg-mgrs.pdf>`__.

   .. code-block:: bash

      conda config --set auto_activate_base false

.. tab-set::

   .. tab-item:: With MPI (only Linux/macOS)

      .. code-block:: bash

         conda create -n warpx-cpu-mpich-dev -c conda-forge blaspp boost ccache cmake compilers git lapackpp "openpmd-api=*=mpi_mpich*" openpmd-viewer python make numpy pandas scipy yt "fftw=*=mpi_mpich*" pkg-config matplotlib mamba mpich mpi4py ninja pip virtualenv
         conda activate warpx-cpu-mpich-dev

         # compile WarpX with -DWarpX_MPI=ON
         # for pip, use: export WARPX_MPI=ON

   .. tab-item:: Without MPI

      .. code-block:: bash

         conda create -n warpx-cpu-dev -c conda-forge blaspp boost ccache cmake compilers git lapackpp openpmd-api openpmd-viewer python make numpy pandas scipy yt fftw pkg-config matplotlib mamba ninja pip virtualenv
         conda activate warpx-cpu-dev

         # compile WarpX with -DWarpX_MPI=OFF
         # for pip, use: export WARPX_MPI=OFF

For OpenMP support, you will further need:

.. tab-set::

   .. tab-item:: Linux

      .. code-block:: bash

         conda install -c conda-forge libgomp

   .. tab-item:: macOS or Windows

      .. code-block:: bash

         conda install -c conda-forge llvm-openmp

For Nvidia CUDA GPU support, you will need to have `a recent CUDA driver installed <https://developer.nvidia.com/cuda-downloads>`__ or you can lower the CUDA version of `the Nvidia cuda package <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation>`__ and `conda-forge to match your drivers <https://docs.cupy.dev/en/stable/install.html#install-cupy-from-conda-forge>`__ and then add these packages:

.. code-block:: bash

   conda install -c nvidia -c conda-forge cuda cuda-nvtx-dev cupy

More info for `CUDA-enabled ML packages <https://twitter.com/jeremyphoward/status/1697435241152127369>`__.


Spack (Linux/macOS)
-------------------

`Spack <https://spack.readthedocs.io>`__ is a user-level package manager.
It is primarily written for Linux, with slightly less support for macOS, and future support for Windows.

First, download a `WarpX Spack desktop development environment <https://github.com/ECP-WarpX/WarpX/blob/development/Tools/machines/desktop>`__ of your choice.
For most desktop developments, pick the OpenMP environment for CPUs unless you have a supported GPU.

* **Debian/Ubuntu** Linux:

  * OpenMP: ``system=ubuntu; compute=openmp`` (CPUs)
  * CUDA: ``system=ubuntu; compute=cuda`` (Nvidia GPUs)
  * ROCm: ``system=ubuntu; compute=rocm`` (AMD GPUs)
  * SYCL: *todo* (Intel GPUs)
* **macOS**: first, prepare with ``brew install gpg2; brew install gcc``

  * OpenMP: ``system=macos; compute=openmp``

If you already `installed Spack <https://spack.io>`__, we recommend to activate its `binary caches <https://spack.io/spack-binary-packages/>`__ for faster builds:

.. code-block:: bash

   spack mirror add rolling https://binaries.spack.io/develop
   spack buildcache keys --install --trust

Now install the WarpX dependencies in a new WarpX development environment:

.. code-block:: bash

   # download environment file
   curl -sLO https://raw.githubusercontent.com/ECP-WarpX/WarpX/development/Tools/machines/desktop/spack-${system}-${compute}.yaml

   # create new development environment
   spack env create warpx-${compute}-dev spack-${system}-${compute}.yaml
   spack env activate warpx-${compute}-dev

   # installation
   spack install
   python3 -m pip install jupyter matplotlib numpy openpmd-api openpmd-viewer pandas scipy virtualenv yt

In new terminal sessions, re-activate the environment with

.. code-block:: bash

   spack env activate warpx-openmp-dev

again.
Replace ``openmp`` with the equivalent you chose.

Compile WarpX with ``-DWarpX_MPI=ON``.
For ``pip``, use ``export WARPX_MPI=ON``.


Brew (macOS/Linux)
------------------

`Homebrew (Brew) <https://brew.sh>`__ is a user-level package manager primarily for `Apple macOS <https://en.wikipedia.org/wiki/MacOS>`__, but also supports Linux.

.. code-block:: bash

   brew update
   brew tap openpmd/openpmd
   brew install adios2      # for openPMD
   brew install ccache
   brew install cmake
   brew install fftw        # for PSATD
   brew install git
   brew install hdf5-mpi    # for openPMD
   brew install libomp
   brew unlink gcc
   brew link --force libomp
   brew install pkg-config  # for fftw
   brew install open-mpi
   brew install openblas    # for PSATD in RZ
   brew install openpmd-api # for openPMD

If you also want to compile with PSATD in RZ, you need to manually install BLAS++ and LAPACK++:

.. code-block:: bash

   sudo mkdir -p /usr/local/bin/
   sudo curl -L -o /usr/local/bin/cmake-easyinstall https://raw.githubusercontent.com/ax3l/cmake-easyinstall/main/cmake-easyinstall
   sudo chmod a+x /usr/local/bin/cmake-easyinstall

   cmake-easyinstall --prefix=/usr/local git+https://github.com/icl-utk-edu/blaspp.git \
       -Duse_openmp=OFF -Dbuild_tests=OFF -DCMAKE_VERBOSE_MAKEFILE=ON
   cmake-easyinstall --prefix=/usr/local git+https://github.com/icl-utk-edu/lapackpp.git \
       -Duse_cmake_find_lapack=ON -Dbuild_tests=OFF -DCMAKE_VERBOSE_MAKEFILE=ON

Compile WarpX with ``-DWarpX_MPI=ON``.
For ``pip``, use ``export WARPX_MPI=ON``.


APT (Debian/Ubuntu Linux)
-------------------------

The `Advanced Package Tool (APT) <https://en.wikipedia.org/wiki/APT_(software)>`__ is a system-level package manager on Debian-based Linux distributions, including Ubuntu.

.. tab-set::

   .. tab-item:: With MPI (only Linux/macOS)

      .. code-block:: bash

         sudo apt update
         sudo apt install build-essential ccache cmake g++ git libfftw3-mpi-dev libfftw3-dev libhdf5-openmpi-dev libopenmpi-dev pkg-config python3 python3-dev python3-matplotlib python3-mpi4py python3-numpy python3-pandas python3-pip python3-scipy python3-venv

         # optional:
         # for CUDA, either install
         #   https://developer.nvidia.com/cuda-downloads (preferred)
         # or, if your Debian/Ubuntu is new enough, use the packages
         #   sudo apt install nvidia-cuda-dev libcub-dev

         # compile WarpX with -DWarpX_MPI=ON
         # for pip, use: export WARPX_MPI=ON

   .. tab-item:: Without MPI

      .. code-block:: bash

         sudo apt update
         sudo apt install build-essential ccache cmake g++ git libfftw3-dev libfftw3-dev libhdf5-dev pkg-config python3 python3-dev python3-matplotlib python3-numpy python3-pandas python3-pip python3-scipy python3-venv

         # optional:
         # for CUDA, either install
         #   https://developer.nvidia.com/cuda-downloads (preferred)
         # or, if your Debian/Ubuntu is new enough, use the packages
         #   sudo apt install nvidia-cuda-dev libcub-dev

         # compile WarpX with -DWarpX_MPI=OFF
         # for pip, use: export WARPX_MPI=OFF
