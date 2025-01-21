.. _building-lawrencium:

Lawrencium (LBNL)
=================

The `Lawrencium cluster <http://scs.lbl.gov/Systems>`_ is located at LBNL.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `Lawrencium user guide <https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/lbnl-supercluster/lawrencium>`_
* Batch system: `Slurm <https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/scheduler/slurm-usage-instructions>`_
* `Production directories <https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/lbnl-supercluster/lawrencium#backup>`_:

  * ``/global/scratch/users/$USER/``: production directory
  * ``/global/home/groups/$GROUP/``: group production directory
  * ``/global/home/users/$USER``: home directory (10 GB)


Installation
------------

Use the following commands to download the WarpX source code and switch to the correct branch:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx

We use the following modules and environments on the system (``$HOME/lawrencium_warpx.profile``).

.. literalinclude:: ../../../../Tools/machines/lawrencium-lbnl/lawrencium_warpx.profile.example
   :language: bash
   :caption: You can copy this file from ``Tools/machines/lawrencium-lbnl/lawrencium_warpx.profile.example``.

We recommend to store the above lines in a file, such as ``$HOME/lawrencium_warpx.profile``, and load it into your shell after a login:

.. code-block:: bash

   source $HOME/lawrencium_warpx.profile

And since Lawrencium does not yet provide a module for them, install ADIOS2, BLAS++ and LAPACK++:

.. code-block:: bash

   # c-blosc (I/O compression)
   git clone -b v1.21.1 https://github.com/Blosc/c-blosc.git src/c-blosc
   rm -rf src/c-blosc-v100-build
   cmake -S src/c-blosc -B src/c-blosc-v100-build -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DDEACTIVATE_AVX2=OFF -DCMAKE_INSTALL_PREFIX=$HOME/sw/v100/c-blosc-1.21.1
   cmake --build src/c-blosc-v100-build --target install --parallel 12

   # ADIOS2
   git clone -b v2.8.3 https://github.com/ornladios/ADIOS2.git src/adios2
   rm -rf src/adios2-v100-build
   cmake -S src/adios2 -B src/adios2-v100-build -DADIOS2_USE_Blosc=ON -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_Python=OFF -DADIOS2_USE_ZeroMQ=OFF -DCMAKE_INSTALL_PREFIX=$HOME/sw/v100/adios2-2.8.3
   cmake --build src/adios2-v100-build --target install -j 12

   # BLAS++ (for PSATD+RZ)
   git clone https://github.com/icl-utk-edu/blaspp.git src/blaspp
   rm -rf src/blaspp-v100-build
   cmake -S src/blaspp -B src/blaspp-v100-build -Duse_openmp=OFF -Dgpu_backend=cuda -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=$HOME/sw/v100/blaspp-master
   cmake --build src/blaspp-v100-build --target install --parallel 12

   # LAPACK++ (for PSATD+RZ)
   git clone https://github.com/icl-utk-edu/lapackpp.git src/lapackpp
   rm -rf src/lapackpp-v100-build
   cmake -S src/lapackpp -B src/lapackpp-v100-build -DCMAKE_CXX_STANDARD=17 -Dgpu_backend=cuda -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=$HOME/sw/v100/lapackpp-master -Duse_cmake_find_lapack=ON -DBLAS_LIBRARIES=${LAPACK_DIR}/lib/libblas.a -DLAPACK_LIBRARIES=${LAPACK_DIR}/lib/liblapack.a
   cmake --build src/lapackpp-v100-build --target install --parallel 12

Optionally, download and install Python packages for :ref:`PICMI <usage-picmi>` or dynamic ensemble optimizations (`libEnsemble <https://libensemble.readthedocs.io/en/main/>`__):

.. code-block:: bash

   python3 -m pip install --user --upgrade pip
   python3 -m pip install --user virtualenv
   python3 -m pip cache purge
   rm -rf $HOME/sw/v100/venvs/warpx
   python3 -m venv $HOME/sw/v100/venvs/warpx
   source $HOME/sw/v100/venvs/warpx/bin/activate
   python3 -m pip install --upgrade pip
   python3 -m pip install --upgrade build
   python3 -m pip install --upgrade packaging
   python3 -m pip install --upgrade wheel
   python3 -m pip install --upgrade setuptools
   python3 -m pip install --upgrade cython
   python3 -m pip install --upgrade numpy
   python3 -m pip install --upgrade pandas
   python3 -m pip install --upgrade scipy
   python3 -m pip install --upgrade mpi4py --no-build-isolation --no-binary mpi4py
   python3 -m pip install --upgrade openpmd-api
   python3 -m pip install --upgrade matplotlib
   python3 -m pip install --upgrade yt
   # optional: for libEnsemble
   python3 -m pip install -r $HOME/src/warpx/Tools/LibEnsemble/requirements.txt

Then, ``cd`` into the directory ``$HOME/src/warpx`` and use the following commands to compile the application executable:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build

   cmake -S . -B build -DWarpX_DIMS="1;2;RZ;3" -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON
   cmake --build build -j 12

The general :ref:`cmake compile-time options <building-cmake>` apply as usual.

**That's it!**
A 3D WarpX executable is now in ``build/bin/`` and :ref:`can be run <running-cpp-lawrencium>` with a :ref:`3D example inputs file <usage-examples>`.
Most people execute the binary directly or copy it out to a location in ``/global/scratch/users/$USER/``.

For a *full PICMI install*, follow the :ref:`instructions for Python (PICMI) bindings <building-cmake-python>`:

.. code-block:: bash

   # PICMI build
   cd $HOME/src/warpx

   # install or update dependencies
   python3 -m pip install -r requirements.txt

   # compile parallel PICMI interfaces in 3D, 2D, 1D and RZ
   WARPX_MPI=ON WARPX_COMPUTE=CUDA WARPX_FFT=ON BUILD_PARALLEL=12 python3 -m pip install --force-reinstall --no-deps -v .

Or, if you are *developing*, do a quick PICMI install of a *single geometry* (see: :ref:`WarpX_DIMS <building-cmake-options>`) using:

.. code-block:: bash

   # find dependencies & configure
   cmake -S . -B build -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS=RZ

   # build and then call "python3 -m pip install ..."
   cmake --build build --target pip_install -j 12


.. _running-cpp-lawrencium:

Running
-------

.. _running-cpp-lawrencium-V100-GPUs:

V100 GPUs (16 GB)
^^^^^^^^^^^^^^^^^

12 nodes with each two NVIDIA V100 GPUs.

.. literalinclude:: ../../../../Tools/machines/lawrencium-lbnl/lawrencium_v100.sbatch
   :language: bash
   :caption: You can copy this file from ``Tools/machines/lawrencium-lbnl/lawrencium_v100.sbatch``.

To run a simulation, copy the lines above to a file ``v100.sbatch`` and run

.. code-block:: bash

   sbatch lawrencium_v100.sbatch

.. _running-cpp-lawrencium-2080Ti-GPUs:

2080 Ti GPUs (10 GB)
^^^^^^^^^^^^^^^^^^^^

18 nodes with each four NVIDIA 2080 TI GPUs.
These are most interesting if you run in single precision.

Use ``--constraint=es1_2080ti --cpus-per-task=2`` in the above template to run on those nodes.
