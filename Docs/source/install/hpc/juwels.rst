.. _building-juwels:

Juwels (JSC)
============

.. note::

   For the moment, WarpX doesn't run on Juwels with MPI_THREAD_MULTIPLE.
   Please compile with this compilation flag: ``MPI_THREAD_MULTIPLE=FALSE``.

The `Juwels supercomputer <https://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUWELS/JUWELS_node.html>`_ is located at JSC.


Introduction
------------

If you are new to this system, **please see the following resources**:

See `this page <https://apps.fz-juelich.de/jsc/hps/juwels/quickintro.html>`_ for a quick introduction.
(Full `user guide <http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUWELS/UserInfo/UserInfo_node.html>`__).

* Batch system: `Slurm <https://apps.fz-juelich.de/jsc/hps/juwels/quickintro.html#batch-system-on-system-name>`__
* `Production directories <https://apps.fz-juelich.de/jsc/hps/juwels/environment.html?highlight=scratch#available-filesystems>`__:

  * ``$SCRATCH``: Scratch filesystem for `temporary data <http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUWELS/FAQ/juwels_FAQ_node.html#faq1495160>`__ (90 day purge)
  * ``$FASTDATA/``: Storage location for large data (backed up)
  * Note that the ``$HOME`` directory is not designed for simulation runs and producing output there will impact performance.


Installation
------------

Use the following commands to download the WarpX source code and switch to the correct branch:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx

We use the following modules and environments on the system.

.. literalinclude:: ../../../../Tools/machines/juwels-jsc/juwels_warpx.profile.example
   :language: bash
   :caption: You can copy this file from ``Tools/machines/juwels-jsc/juwels_warpx.profile.example``.

Note that for now WarpX must rely on OpenMPI instead of the recommended MPI implementation on this platform MVAPICH2.

We recommend to store the above lines in a file, such as ``$HOME/juwels_warpx.profile``, and load it into your shell after a login:

.. code-block:: bash

   source $HOME/juwels_warpx.profile

Then, ``cd`` into the directory ``$HOME/src/warpx`` and use the following commands to compile:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build

   cmake -S . -B build -DWarpX_DIMS="1;2;3" -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_MPI_THREAD_MULTIPLE=OFF
   cmake --build build -j 16

The other :ref:`general compile-time options <install-developers>` apply as usual.

**That's it!**
A 3D WarpX executable is now in ``build/bin/`` and :ref:`can be run <running-cpp-juwels>` with a :ref:`3D example inputs file <usage-examples>`.
Most people execute the binary directly or copy it out to a location in ``$SCRATCH``.

.. note::

   Currently, if you want to use HDF5 output with openPMD, you need to add

   .. code-block:: bash

      export OMPI_MCA_io=romio321

   in your job scripts, before running the ``srun`` command.

.. _running-cpp-juwels:

Running
-------

Queue: gpus (4 x Nvidia V100 GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Juwels GPUs <https://apps.fz-juelich.de/jsc/hps/juwels/configuration.html>`__ are V100 (16GB) and A100 (40GB).

An example submission script reads

.. literalinclude:: ../../../../Tools/machines/juwels-jsc/juwels.sbatch
   :language: bash
   :caption: You can copy this file from ``Tools/machines/juwels-jsc/juwels.sbatch``.

Queue: batch (2 x Intel Xeon Platinum 8168 CPUs, 24 Cores + 24 Hyperthreads/CPU)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*todo*

See the :ref:`data analysis section <dataanalysis-formats>` for more information on how to visualize the simulation results.
