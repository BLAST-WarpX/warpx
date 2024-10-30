.. _building-lonestar6:

Lonestar6 (TACC)
================

The `Lonestar6 cluster <https://portal.tacc.utexas.edu/user-guides/lonestar6>`_ is located at `TACC <https://www.tacc.utexas.edu>`__.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `TACC user guide <https://portal.tacc.utexas.edu/user-guides/>`__
* Batch system: `Slurm <https://portal.tacc.utexas.edu/user-guides/lonestar6#job-management>`__
* `Jupyter service <https://tacc.github.io/ctls2017/docs/intro_to_python/intro_to_python_011_jupyter.html>`__
* `Filesystem directories <https://portal.tacc.utexas.edu/user-guides/lonestar6#managing-files-on-lonestar6>`__:

  * ``$HOME``: per-user home directory, backed up (10 GB)
  * ``$WORK``: per-user production directory, not backed up, not purged, Lustre (1 TB)
  * ``$SCRATCH``: per-user production directory, not backed up, purged every 10 days, Lustre (no limits, 8PByte total)


Installation
------------

Use the following commands to download the WarpX source code and switch to the correct branch:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $WORK/src/warpx

We use system software modules, add environment hints and further dependencies via the file ``$HOME/lonestar6_warpx_a100.profile``.
Create it now:

.. code-block:: bash

   cp $HOME/src/warpx/Tools/machines/lonestar6-tacc/lonestar6_warpx_a100.profile.example $HOME/lonestar6_warpx_a100.profile

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/lonestar6-tacc/lonestar6_warpx_a100.profile.example
      :language: bash

Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
For example, if you are member of the project ``abcde``, then run ``nano $HOME/lonestar6_warpx_a100.profile`` and edit line 2 to read:

.. code-block:: bash

   export proj="abcde"

Exit the ``nano`` editor with ``Ctrl`` + ``O`` (save) and then ``Ctrl`` + ``X`` (exit).

.. important::

   Now, and as the first step on future logins to Lonestar6, activate these environment settings:

   .. code-block:: bash

      source $HOME/lonestar6_warpx_a100.profile

Finally, since Lonestar6 does not yet provide software modules for some of our dependencies, install them once:

.. code-block:: bash

   bash $HOME/src/warpx/Tools/machines/lonestar6-tacc/install_a100_dependencies.sh
   source ${SW_DIR}/venvs/warpx-a100/bin/activate

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/lonestar6-tacc/install_a100_dependencies.sh
      :language: bash


.. _building-lonestar6-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build_pm_gpu

   cmake -S . -B build_gpu -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_gpu -j 16

The WarpX application executables are now in ``$HOME/src/warpx/build_gpu/bin/``.
Additionally, the following commands will install WarpX as a Python module:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build_pm_gpu_py

   cmake -S . -B build_gpu_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_gpu_py -j 16 --target pip_install

Now, you can :ref:`submit Lonestar6 compute jobs <running-cpp-lonestar6>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit Lonestar6 jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-cpp-lonestar6>` or copy them to a location in ``$WORK`` or ``$SCRATCH``.


.. _running-cpp-lonestar6:

Running
-------

.. _running-cpp-lonestar6-A100-GPUs:

A100 GPUs (40 GB)
^^^^^^^^^^^^^^^^^

`84 GPU nodes, each with 2 A100 GPUs (40 GB) <https://portal.tacc.utexas.edu/user-guides/lonestar6#system-gpu>`__.

The batch script below can be used to run a WarpX simulation on multiple nodes (change ``-N`` accordingly) on the supercomputer lonestar6 at tacc.
Replace descriptions between chevrons ``<>`` by relevant values, for instance ``<input file>`` could be ``plasma_mirror_inputs``.
Note that we run one MPI rank per GPU.


.. literalinclude:: ../../../../Tools/machines/lonestar6-tacc/lonestar6_a100.sbatch
   :language: bash
   :caption: You can copy this file from ``Tools/machines/lonestar6-tacc/lonestar6_a100.sbatch``.

To run a simulation, copy the lines above to a file ``lonestar6.sbatch`` and run

.. code-block:: bash

   sbatch lonestar6_a100.sbatch

to submit the job.
