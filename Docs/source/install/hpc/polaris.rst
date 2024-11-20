.. _building-polaris:

Polaris (ALCF)
==============

The `Polaris cluster <https://docs.alcf.anl.gov/polaris/getting-started/>`__ is located at ALCF.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `ALCF user guide <https://docs.alcf.anl.gov/>`__
* Batch system: `PBS <https://docs.alcf.anl.gov/running-jobs/job-and-queue-scheduling/>`__
* `Filesystems <https://docs.alcf.anl.gov/data-management/filesystem-and-storage/file-systems/>`__

.. _building-polaris-preparation:

Preparation
-----------

Use the following commands to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx

On Polaris, you can run either on GPU nodes with fast A100 GPUs (recommended) or CPU nodes.

.. tab-set::

   .. tab-item:: A100 GPUs

      We use system software modules, add environment hints and further dependencies via the file ``$HOME/polaris_gpu_warpx.profile``.
      Create it now:

      .. code-block:: bash

         cp $HOME/src/warpx/Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example $HOME/polaris_gpu_warpx.profile

      .. dropdown:: Script Details
         :color: light
         :icon: info
         :animate: fade-in-slide-down

         .. literalinclude:: ../../../../Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example
            :language: bash

      Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
      For example, if you are member of the project ``proj_name``, then run ``nano $HOME/polaris_gpu_warpx.profile`` and edit line 2 to read:

      .. code-block:: bash

         export proj="proj_name"

      Exit the ``nano`` editor with ``Ctrl`` + ``O`` (save) and then ``Ctrl`` + ``X`` (exit).

      .. important::

         Now, and as the first step on future logins to Polaris, activate these environment settings:

         .. code-block:: bash

            source $HOME/polaris_gpu_warpx.profile

      Finally, since Polaris does not yet provide software modules for some of our dependencies, install them once:

      .. code-block:: bash

         bash $HOME/src/warpx/Tools/machines/polaris-alcf/install_gpu_dependencies.sh
         source ${CFS}/${proj%_g}/${USER}/sw/polaris/gpu/venvs/warpx/bin/activate

      .. dropdown:: Script Details
         :color: light
         :icon: info
         :animate: fade-in-slide-down

         .. literalinclude:: ../../../../Tools/machines/polaris-alcf/install_gpu_dependencies.sh
            :language: bash


   .. tab-item:: CPU Nodes

      *Under construction*


.. _building-polaris-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. tab-set::

   .. tab-item:: A100 GPUs

      .. code-block:: bash

         cd $HOME/src/warpx
         rm -rf build_pm_gpu

         cmake -S . -B build_pm_gpu -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
         cmake --build build_pm_gpu -j 16

      The WarpX application executables are now in ``$HOME/src/warpx/build_pm_gpu/bin/``.
      Additionally, the following commands will install WarpX as a Python module:

      .. code-block:: bash

         cd $HOME/src/warpx
         rm -rf build_pm_gpu_py

         cmake -S . -B build_pm_gpu_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
         cmake --build build_pm_gpu_py -j 16 --target pip_install

   .. tab-item:: CPU Nodes

      *Under construction*

Now, you can :ref:`submit Polaris compute jobs <running-cpp-polaris>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit Polaris jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-cpp-polaris>` or copy them to a location in ``$PSCRATCH``.


.. _building-polaris-update:

Update WarpX & Dependencies
---------------------------

If you already installed WarpX in the past and want to update it, start by getting the latest source code:

.. code-block:: bash

   cd $HOME/src/warpx

   # read the output of this command - does it look ok?
   git status

   # get the latest WarpX source code
   git fetch
   git pull

   # read the output of these commands - do they look ok?
   git status
   git log # press q to exit

And, if needed,

- :ref:`update the polaris_gpu_warpx.profile or polaris_cpu_warpx files <building-polaris-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-polaris-preparation>`.

As a last step, clean the build directory ``rm -rf $HOME/src/warpx/build_pm_*`` and rebuild WarpX.


.. _running-cpp-polaris:

Running
-------

.. tab-set::

   .. tab-item:: A100 (40GB) GPUs

      The batch script below can be used to run a WarpX simulation on multiple nodes (change ``<NODES>`` accordingly) on the supercomputer Polaris at ALCF.

      Replace descriptions between chevrons ``<>`` by relevant values, for instance ``<input file>`` could be ``plasma_mirror_inputs``.
      Note that we run one MPI rank per GPU.

      .. literalinclude:: ../../../../Tools/machines/polaris-alcf/polaris_gpu.pbs
         :language: bash
         :caption: You can copy this file from ``$HOME/src/warpx/Tools/machines/polaris-alcf/polaris_gpu.pbs``.

      To run a simulation, copy the lines above to a file ``polaris_gpu.pbs`` and run

      .. code-block:: bash

         qsub polaris_gpu.pbs

      to submit the job.


   .. tab-item:: CPU Nodes

      *Under construction*
