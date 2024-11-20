.. _developers-gnumake:

GNUmake Build System (Legacy)
=============================

:ref:`CMake <install-developers>` is our primary build system.
In this section, we describe our legacy build scripts - do not use them unless you used them before.

WarpX is built on `AMReX <https://amrex-codes.github.io>`__, which also provides support for a Linux-centric set of build scripts implemented in GNUmake.
Since we sometimes need to move fast and test highly experimental compilers and Unix derivates on core components of WarpX, this set of build scripts is used by some of our experienced developers.

.. warning::

   On the long-term, these scripts do not scale to the full feature set of WarpX and its dependencies.
   Please see the CMake-based :ref:`developer section <install-developers>` instead.

This page describes the most basic build with GNUmake files and points to instructions for more advanced builds.

Downloading the source code
---------------------------

Clone the source codes of WarpX, and its dependencies AMReX and PICSAR into one
single directory (e.g. ``warpx_directory``):

::

    mkdir warpx_directory
    cd warpx_directory
    git clone https://github.com/ECP-WarpX/WarpX.git
    git clone https://github.com/ECP-WarpX/picsar.git
    git clone https://github.com/ECP-WarpX/warpx-data.git
    git clone https://github.com/AMReX-Codes/amrex.git

.. note::
   The warpx-data repository is currently only needed for MCC cross-sections.

Basic compilation
-----------------

WarpX requires a C/C++ compiler (e.g., GNU, LLVM or Intel) and an MPI implementation (e.g., OpenMPI or MPICH).
Start a GNUmake build by ``cd``-ing into the directory ``WarpX`` and type

::

    make -j 4

This will generate an executable file in the ``Bin`` directory.

Compile-time vs. run-time options
---------------------------------

WarpX has multiple compile-time and run-time options. The compilation
options are set in the file ``GNUmakefile``. The default
options correspond to an optimized code for 3D geometry. The main compile-time
options are:

    * ``DIM=3`` or ``2``: Geometry of the simulation (note that running an executable compiled for 3D with a 2D input file will crash).
    * ``DEBUG=FALSE`` or ``TRUE``: Compiling in ``DEBUG`` mode can help tremendously during code development.
    * ``USE_FFT=FALSE`` or ``TRUE``: Compile the Pseudo-Spectral Analytical Time Domain Maxwell solver. Requires an FFT library.
    * ``USE_RZ=FALSE`` or ``TRUE``: Compile for 2D axisymmetric geometry.
    * ``COMP=gcc`` or ``intel``: Compiler.
    * ``USE_MPI=TRUE`` or ``FALSE``: Whether to compile with MPI support.
    * ``USE_OMP=TRUE`` or ``FALSE``: Whether to compile with OpenMP support.
    * ``USE_GPU=TRUE`` or ``FALSE``: Whether to compile for Nvidia GPUs (requires CUDA).
    * ``USE_OPENPMD=TRUE`` or ``FALSE``: Whether to support openPMD for I/O (requires openPMD-api).
    * ``MPI_THREAD_MULTIPLE=TRUE`` or ``FALSE``: Whether to initialize MPI with thread multiple support. Required to use asynchronous IO with more than ``amrex.async_out_nfiles`` (by default, 64) MPI tasks.
      Please see :ref:`data formats <dataanalysis-formats>` for more information.
    * ``PRECISION=FLOAT USE_SINGLE_PRECISION_PARTICLES=TRUE``: Switch from default double precision to single precision (experimental).

For a description of these different options, see the `corresponding page <https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html>`__ in the AMReX documentation.

Alternatively, instead of modifying the file ``GNUmakefile``, you can directly pass the options in command line ; for instance:

::

   make -j 4 USE_OMP=FALSE

In order to clean a previously compiled version (typically useful for troubleshooting, if you encounter unexpected compilation errors):

::

    make realclean

before re-attempting compilation.

Advanced GNUmake instructions
-----------------------------

.. toctree::
   :maxdepth: 1

   gnumake/openpmd
   gnumake/spectral
   gnumake/rzgeometry
   gnumake/gpu_local
   gnumake/python
   gnumake/spack
