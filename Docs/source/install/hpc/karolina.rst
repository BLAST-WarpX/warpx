.. _building-karolina:

Karolina (IT4I)
===============

The `Karolina cluster <https://docs.it4i.cz/karolina/introduction/>`_ is located at `IT4I, Technical University of Ostrava <https://www.it4i.cz/en>`__.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `IT4I user guide <https://docs.it4i.cz>`__
* Batch system: `SLURM <https://docs.it4i.cz/general/job-submission-and-execution/>`__
* Jupyter service: not provided/documented (yet)
* `Filesystems <https://docs.it4i.cz/karolina/storage/>`__:

  * ``$HOME``: per-user directory, use only for inputs, source and scripts; backed up (25GB default quota)
  * ``/scratch/``: `production directory <https://docs.it4i.cz/karolina/storage/#scratch-file-system>`__; very fast for parallel jobs (10TB default)
  * ``/mnt/proj<N>/<proj>``: per-project work directory, used for long term data storage (20TB default)


.. _building-karolina-preparation:

Installation
------------

We show how to install from scratch all the dependencies using `Spack <https://spack.io>`__.

For size reasons it is not advisable to install WarpX in the ``$HOME`` directory, it should be installed in the "work directory". For this purpose we set an environment variable ``$WORK`` with the path to the "work directory".

On Karolina, you can run either on GPU nodes with fast A100 GPUs (recommended) or CPU nodes.

Profile file
^^^^^^^^^^^^

One can use the pre-prepared ``karolina_warpx.profile`` script below,
which you can copy to ``${HOME}/karolina_warpx.profile``, edit as required and then ``source``.

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/karolina-it4i/karolina_warpx.profile.example
      :language: bash
      :caption: Copy the contents of this file to ``${HOME}/karolina_warpx.profile``.

To have the environment activated on every login, add the following line to ``${HOME}/.bashrc``:

.. code-block:: bash

    source $HOME/karolina_warpx.profile

To install the ``spack`` environment and Python packages:

.. code-block:: bash

   bash $WORK/src/warpx/Tools/machines/karolina-it4i/install_dependencies.sh

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/karolina-it4i/install_dependencies.sh
      :language: bash


.. _building-karolina-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. code-block:: bash

   cd $WORK/src/warpx
   rm -rf build_gpu

   cmake -S . -B build_gpu -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_gpu -j 48

The WarpX application executables are now in ``$WORK/src/warpx/build_gpu/bin/``.
Additionally, the following commands will install WarpX as a Python module:

.. code-block:: bash

   cd $WORK/src/warpx
   rm -rf build_gpu_py

   cmake -S . -B build_gpu_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_gpu_py -j 48 --target pip_install

Now, you can :ref:`submit Karolina compute jobs <running-cpp-karolina>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit Karolina jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-cpp-karolina>` or copy them to a location in ``/scratch/``.


.. _running-cpp-karolina:

Running
-------

The batch script below can be used to run a WarpX simulation on multiple GPU nodes (change ``#SBATCH --nodes=`` accordingly) on the supercomputer Karolina at IT4I.
This partition has up to `72 nodes <https://docs.it4i.cz/karolina/hardware-overview/>`__.
Every node has 8x A100 (40GB) GPUs and 2x AMD EPYC 7763, 64-core, 2.45 GHz processors.

Replace descriptions between chevrons ``<>`` by relevant values, for instance ``<proj>`` could be ``DD-23-83``.
Note that we run one MPI rank per GPU.

.. literalinclude:: ../../../../Tools/machines/karolina-it4i/karolina_gpu.sbatch
   :language: bash
   :caption: You can copy this file from ``$WORK/src/warpx/Tools/machines/karolina-it4i/karolina_gpu.sbatch``.

To run a simulation, copy the lines above to a file ``karolina_gpu.sbatch`` and run

.. code-block:: bash

   sbatch karolina_gpu.sbatch

to submit the job.


.. _post-processing-karolina:

Post-Processing
---------------

.. note::

   This section was not yet written.
   Usually, we document here how to use a Jupyter service.
