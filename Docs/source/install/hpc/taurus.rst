.. _building-taurus:

Taurus (ZIH)
============

The `Taurus cluster <https://doc.zih.tu-dresden.de/jobs_and_resources/overview>`_ is located at `ZIH (TU Dresden) <https://doc.zih.tu-dresden.de>`__.

The cluster has multiple partitions, this section describes how to use the `AMD Rome CPUs + NVIDIA A100¶ <https://doc.zih.tu-dresden.de/jobs_and_resources/hardware_overview/#amd-rome-cpus-nvidia-a100>`__.

Introduction
------------

If you are new to this system, **please see the following resources**:

* `ZIH user guide <https://docs.nersc.gov/>`__
* Batch system: `Slurm <https://docs.nersc.gov/systems/perlmutter/#running-jobs>`__
* Jupyter service: Missing?
* `Production directories <https://docs.nersc.gov/filesystems/perlmutter-scratch/>`__:

  * ``$PSCRATCH``: per-user production directory, purged every 30 days (<TBD>TB)
  * ``/global/cscratch1/sd/m3239``: shared production directory for users in the project ``m3239``, purged every 30 days (50TB)
  * ``/global/cfs/cdirs/m3239/``: community file system for users in the project ``m3239`` (100TB)


Installation
------------

Use the following commands to download the WarpX source code and switch to the correct branch:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx

We use the following modules and environments on the system (``$HOME/taurus_warpx.profile``).

.. literalinclude:: ../../../../Tools/machines/taurus-zih/taurus_warpx.profile.example
   :language: bash
   :caption: You can copy this file from ``Tools/machines/taurus-zih/taurus_warpx.profile.example``.

We recommend to store the above lines in a file, such as ``$HOME/taurus_warpx.profile``, and load it into your shell after a login:

.. code-block:: bash

   source $HOME/taurus_warpx.profile

Then, ``cd`` into the directory ``$HOME/src/warpx`` and use the following commands to compile:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build

   cmake -S . -B build -DWarpX_DIMS="1;2;3" -DWarpX_COMPUTE=CUDA
   cmake --build build -j 16

The general :ref:`cmake compile-time options <building-cmake>` apply as usual.


.. _running-cpp-taurus:

Running
-------

.. _running-cpp-taurus-A100-GPUs:

A100 GPUs (40 GB)
^^^^^^^^^^^^^^^^^

The `alpha` partition has 34 nodes with 8 x NVIDIA A100-SXM4 Tensor Core-GPUs and 2 x AMD EPYC CPU 7352 (24 cores) @ 2.3 GHz (multithreading disabled) per node.

The batch script below can be used to run a WarpX simulation on multiple nodes (change ``-N`` accordingly).
Replace descriptions between chevrons ``<>`` by relevant values, for instance ``<input file>`` could be ``plasma_mirror_inputs``.
Note that we run one MPI rank per GPU.


.. literalinclude:: ../../../../Tools/machines/taurus-zih/taurus.sbatch
   :language: bash
   :caption: You can copy this file from ``Tools/machines/taurus-zih/taurus.sbatch``.

To run a simulation, copy the lines above to a file ``taurus.sbatch`` and run

.. code-block:: bash

   sbatch taurus.sbatch

to submit the job.
