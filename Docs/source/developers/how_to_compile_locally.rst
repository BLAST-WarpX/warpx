.. _developers-local-compile:

How to compile locally and fast
===============================

For simplicity, WarpX :ref:`compilation with CMake <building-cmake>` by default downloads, configures and compiles compatible versions of :ref:`central dependencies <install-dependencies>` such as:

* `AMReX <https://amrex-codes.github.io>`__
* `PICSAR <https://github.com/ECP-WarpX/picsar>`__
* `openPMD-api <https://github.com/openPMD/openPMD-api>`__
* `pyAMReX <https://github.com/AMReX-Codes/pyamrex>`__
* `pybind11 <https://github.com/pybind/pybind11>`__

on-the-fly, which is called a *superbuild*.

In some scenarios, e.g., when compiling without internet, with slow internet access, or when working on WarpX and its dependencies, modifications to the superbuild strategy might be preferable.
In the below workflows, you as the developer need to make sure to use compatible versions of the dependencies you provide.


.. _developers-local-compile-src:

Compiling From Local Sources
----------------------------

This workflow is best for developers that make changes to WarpX, AMReX, PICSAR, openPMD-api and/or pyAMReX at the same time.
For instance, use this if you add a feature in AMReX and want to try it in WarpX before it is proposed as a pull request for inclusion in AMReX.

Instead of downloading the source code of the above dependencies, one can also use an already cloned source copy.
For instance, clone these dependencies to ``$HOME/src``:

.. code-block:: bash

   cd $HOME/src

   git clone https://github.com/ECP-WarpX/WarpX.git warpx
   git clone https://github.com/AMReX-Codes/amrex.git
   git clone https://github.com/openPMD/openPMD-api.git
   git clone https://github.com/ECP-WarpX/picsar.git
   git clone https://github.com/AMReX-Codes/pyamrex.git
   git clone https://github.com/pybind/pybind11.git

Now modify the dependencies as needed in their source locations, update sources if you cloned them earlier, etc.
When building WarpX, :ref:`the following CMake flags <building-cmake-options>` will use the respective local sources:

.. code-block:: bash

   cd src/warpx

   rm -rf build

   cmake -S . -B build  \
     -DWarpX_PYTHON=ON  \
     -DWarpX_amrex_src=$HOME/src/amrex          \
     -DWarpX_openpmd_src=$HOME/src/openPMD-api  \
     -DWarpX_picsar_src=$HOME/src/picsar        \
     -DWarpX_pyamrex_src=$HOME/src/pyamrex      \
     -DWarpX_pybind11_src=$HOME/src/pybind11

   cmake --build build -j 8
   cmake --build build -j 8 --target pip_install


.. _developers-local-compile-findpackage:

Compiling With Pre-Compiled Dependencies
----------------------------------------

This workflow is the best and fastest to compile WarpX, when you just want to change code in WarpX and have the above central dependencies already made available *in the right configurations* (e.g., w/ or w/o MPI or GPU support) from a :ref:`module system <install-hpc>` or :ref:`package manager <install-dependencies>`.

Instead of downloading the source code of the above central dependencies, or using a local copy of their source, we can compile and install those dependencies once.
By setting the `CMAKE_PREFIX_PATH <https://cmake.org/cmake/help/latest/envvar/CMAKE_PREFIX_PATH.html>`__ environment variable to the respective dependency install location prefixes, we can instruct CMake to `find their install locations and configurations <https://hsf-training.github.io/hsf-training-cmake-webpage/09-findingpackages/index.html>`__.

WarpX supports this with :ref:`the following CMake flags <building-cmake-options>`:

.. code-block:: bash

   cd src/warpx

   rm -rf build

   cmake -S . -B build  \
     -DWarpX_PYTHON=ON  \
     -DWarpX_amrex_internal=OFF    \
     -DWarpX_openpmd_internal=OFF  \
     -DWarpX_picsar_internal=OFF   \
     -DWarpX_pyamrex_internal=OFF  \
     -DWarpX_pybind11_internal=OFF

   cmake --build build -j 8
   cmake --build build -j 8 --target pip_install

As a background, this is also the workflow how WarpX is built in :ref:`package managers such as Spack and Conda-Forge <install-dependencies>`.


.. _developers-local-compile-pylto:

Faster Python Builds
--------------------

The Python bindings of WarpX and AMReX (pyAMReX) use `pybind11 <https://pybind11.readthedocs.io>`__.
Since pybind11 relies heavily on `C++ metaprogramming <https://pybind11.readthedocs.io/en/stable/faq.html#how-can-i-create-smaller-binaries>`__, speeding up the generated binding code requires that we perform a `link-time optimization (LTO) <https://pybind11.readthedocs.io/en/stable/compiling.html#pybind11-add-module>`__ step, also known as `interprocedural optimization (IPO) <https://en.wikipedia.org/wiki/Interprocedural_optimization>`__.

For fast local development cycles, one can skip LTO/IPO with the following flags:

.. code-block:: bash

   cd src/warpx

   cmake -S . -B build       \
     -DWarpX_PYTHON=ON       \
     -DWarpX_PYTHON_IPO=OFF  \
     -DpyAMReX_IPO=OFF

   cmake --build build -j 8 --target pip_install

.. note::

   We might transition to `nanobind <https://github.com/wjakob/nanobind>`__ in the future, which `does not rely on LTO/IPO <https://nanobind.readthedocs.io/en/latest/benchmark.html>`__ for optimal binaries.
   You can contribute to `this pyAMReX pull request <https://github.com/AMReX-Codes/pyamrex/pull/127>`__ to help exploring this library (and if it works for the HPC/GPU compilers that we need to support).

For robustness, our ``pip_install`` target performs a regular ``wheel`` build and then installs it with ``pip``.
This step will check every time of WarpX dependencies are properly installed, to avoid broken installations.
When developing without internet or after the first ``pip_install`` succeeded in repeated installations in rapid development cycles, this check of ``pip`` can be skipped by using the ``pip_install_nodeps`` target instead:

.. code-block:: bash

   cmake --build build -j 8 --target pip_install_nodeps


.. _developers-local-compile-ccache:

CCache
------

WarpX builds will automatically search for `CCache <https://ccache.dev>`__ to speed up subsequent compilations in development cycles.
Make sure a :ref:`recent CCache version <install-dependencies>` is installed to make use of this feature.

For power developers that switch a lot between fundamentally different WarpX configurations (e.g., 1D to 3D, GPU and CPU builds, many branches with different bases, developing AMReX and WarpX at the same time), also consider increasing the `CCache cache size <https://ccache.dev/manual/4.9.html#_cache_size_management>`__ and changing the `cache directory <https://ccache.dev/manual/4.9.html#config_cache_dir>`__ if needed, e.g., due to storage quota constraints or to choose a fast(er) filesystem for the cache files.
