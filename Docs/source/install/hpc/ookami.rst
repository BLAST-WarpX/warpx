.. _building-ookami:

Ookami (Stony Brook)
====================

The `Ookami cluster <https://www.stonybrook.edu/ookami/>`__ is located at Stony Brook University.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `Ookami documentation <https://www.stonybrook.edu/commcms/ookami/support/index_links_and_docs.php>`__
* Batch system: `Slurm <https://www.stonybrook.edu/commcms/ookami/support/faq/example-slurm-script>`__ (see `available queues <https://www.stonybrook.edu/commcms/ookami/support/faq/queues_on_ookami>`__)
* `Filesystem locations <https://www.stonybrook.edu/commcms/ookami/support/faq/ookami_storage_options.php>`__:

  * ``/lustre/home/<netid>`` (30GByte, backuped)
  * ``/lustre/scratch/<netid>`` (14 day purge)
  * ``/lustre/projects/<your_group>*`` (1TByte default, up to 8TB possible, shared within our group/project, backuped, prefer this location)

We use Ookami as a development cluster for `A64FX <https://www.arm.com/blogs/blueprint/fujitsu-a64fx-arm>`__,
The cluster also provides a few extra nodes, e.g. two ``Thunder X2`` (ARM) nodes.



Installation
------------

Use the following commands to download the WarpX source code and switch to the correct branch:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx

We use the following modules and environments on the system (``$HOME/warpx_gcc10.profile``).

.. literalinclude:: ../../../../Tools/machines/ookami-sbu/ookami_warpx.profile.example
   :language: bash
   :caption: You can copy this file from ``Tools/machines/ookami-sbu/ookami_warpx.profile.example``.

We recommend to store the above lines in a file, such as ``$HOME/warpx_gcc10.profile``, and load it into your shell after a login:

.. code-block:: bash

   source $HOME/warpx_gcc10.profile


Then, ``cd`` into the directory ``$HOME/src/warpx`` and use the following commands to compile:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build

   cmake -S . -B build -DWarpX_COMPUTE=OMP -DWarpX_DIMS="1;2;3"
   cmake --build build -j 10

   # or (currently better performance)
   cmake -S . -B build -DWarpX_COMPUTE=NOACC -DWarpX_DIMS="1;2;3"
   cmake --build build -j 10

The general :ref:`cmake compile-time options <building-cmake>` apply as usual.

**That's it!**
A 3D WarpX executable is now in ``build/bin/`` and :ref:`can be run <running-cpp-ookami>` with a :ref:`3D example inputs file <usage-examples>`.
Most people execute the binary directly or copy it out to a location in ``/lustre/scratch/<netid>``.


.. _running-cpp-ookami:

Running
-------

For running on 48 cores of a single node:

.. code-block:: bash

   srun -p short -N 1 -n 48 --pty bash
   OMP_NUM_THREADS=1 mpiexec -n 48 --map-by ppr:12:numa:pe=1 --report-bindings ./warpx inputs

   # alternatively, using 4 MPI ranks with each 12 threads on a single node:
   OMP_NUM_THREADS=12 mpiexec -n 4 --map-by ppr:4:numa:pe=12 --report-bindings ./warpx inputs

The Ookami HPE Apollo 80 system has 174 A64FX compute nodes each with 32GB of high-bandwidth memory.


Additional Compilers
--------------------

This section is just a note for developers.
We compiled with the Fujitsu Compiler (Clang) with the following build string:

.. code-block:: bash

   cmake -S . -B build                              \
      -DCMAKE_C_COMPILER=$(which mpifcc)            \
      -DCMAKE_C_COMPILER_ID="Clang"                 \
      -DCMAKE_C_COMPILER_VERSION=12.0               \
      -DCMAKE_C_STANDARD_COMPUTED_DEFAULT="11"      \
      -DCMAKE_CXX_COMPILER=$(which mpiFCC)          \
      -DCMAKE_CXX_COMPILER_ID="Clang"               \
      -DCMAKE_CXX_COMPILER_VERSION=12.0             \
      -DCMAKE_CXX_STANDARD_COMPUTED_DEFAULT="14"    \
      -DCMAKE_CXX_FLAGS="-Nclang"                   \
      -DAMReX_DIFFERENT_COMPILER=ON                 \
      -DAMReX_MPI_THREAD_MULTIPLE=FALSE             \
      -DWarpX_COMPUTE=OMP
   cmake --build build -j 10

Note that the best performance for A64FX is currently achieved with the GCC or ARM compilers.
