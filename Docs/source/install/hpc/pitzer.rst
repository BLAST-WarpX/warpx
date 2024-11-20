.. _building-pitzer:

Pitzer (OSC)
============

The `Pitzer cluster <https://www.osc.edu/supercomputing/computing/pitzer>`__ is located at the Ohio Supercomputer Center (OSC). It is currently the main CPU/GPU cluster at OSC. However, the `Cardinal cluster <https://www.osc.edu/resources/technical_support/supercomputers/cardinal>`__ is soon going to take over Pitzer to become the next major CPU/GPU cluster at OSC in the second half of 2024. A list of all OSC clusters can be found `here <https://www.osc.edu/services/cluster_computing>`__.

The Pitzer cluster offers a variety of partitions suitable for different computational needs, including GPU nodes, CPU nodes, and nodes with large memory capacities. For more information on the specifications and capabilities of these partitions, visit the `Ohio Supercomputer Center's Pitzer page <https://www.osc.edu/supercomputing/computing/pitzer>`__.

Introduction
------------

If you are new to this system, **please see the following resources**:

* `Pitzer user guide <https://www.osc.edu/resources/getting_started/new_user_resource_guide>`__
* Batch system: `Slurm <https://www.osc.edu/supercomputing/batch-processing-at-osc>`__
* `Jupyter service <https://www.osc.edu/vocabulary/documentation/jupyter>`__
* `Filesystems <https://www.osc.edu/supercomputing/storage-environment-at-osc/storage-hardware/overview_of_file_systems>`__:

  * ``$HOME``: per-user directory, use only for inputs, source, and scripts; backed up (500GB)
  * ``/fs/ess``: per-project storage directory, use for long-term storage of data and analysis; backed up (1-5TB)
  * ``/fs/scratch``: per-project production directory; fast I/O for parallel jobs; not backed up (100TB)

.. _building-pitzer-preparation:

Preparation
-----------

Use the following commands to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx

On Pitzer, you can run either on GPU nodes with V100 GPUs or CPU nodes.

.. tab-set::

   .. tab-item:: V100 GPUs

      We use system software modules, add environment hints and further dependencies via the file ``$HOME/pitzer_v100_warpx.profile``.
      Create it now:

      .. code-block:: bash

         cp $HOME/src/warpx/Tools/machines/pitzer-osc/pitzer_v100_warpx.profile.example $HOME/pitzer_v100_warpx.profile

      .. dropdown:: Script Details
         :color: light
         :icon: info
         :animate: fade-in-slide-down

         .. literalinclude:: ../../../../Tools/machines/pitzer-osc/pitzer_v100_warpx.profile.example
            :language: bash

      Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
      For example, if you are member of the project ``pas2024``, then run ``nano $HOME/pitzer_v100_warpx.profile`` and edit line 2 to read:

      .. code-block:: bash

         export proj="pas2024"

      Exit the ``nano`` editor with ``Ctrl`` + ``O`` (save) and then ``Ctrl`` + ``X`` (exit).

      .. important::

         Now, and as the first step on future logins to pitzer, activate these environment settings:

         .. code-block:: bash

            source $HOME/pitzer_v100_warpx.profile

      Finally, since pitzer does not yet provide software modules for some of our dependencies, install them once:

      .. code-block:: bash

         bash $HOME/src/warpx/Tools/machines/pitzer-osc/install_v100_dependencies.sh

      .. dropdown:: Script Details
         :color: light
         :icon: info
         :animate: fade-in-slide-down

         .. literalinclude:: ../../../../Tools/machines/pitzer-osc/install_v100_dependencies.sh
            :language: bash


   .. tab-item:: CPU Nodes

      We use system software modules, add environment hints and further dependencies via the file ``$HOME/pitzer_cpu_warpx.profile``.
      Create it now:

      .. code-block:: bash

         cp $HOME/src/warpx/Tools/machines/pitzer-osc/pitzer_cpu_warpx.profile.example $HOME/pitzer_cpu_warpx.profile

      .. dropdown:: Script Details
         :color: light
         :icon: info
         :animate: fade-in-slide-down

         .. literalinclude:: ../../../../Tools/machines/pitzer-osc/pitzer_cpu_warpx.profile.example
            :language: bash

      Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
      For example, if you are member of the project ``pas2024``, then run ``nano $HOME/pitzer_cpu_warpx.profile`` and edit line 2 to read:

      .. code-block:: bash

         export proj="pas2024"

      Exit the ``nano`` editor with ``Ctrl`` + ``O`` (save) and then ``Ctrl`` + ``X`` (exit).

      .. important::

         Now, and as the first step on future logins to pitzer, activate these environment settings:

         .. code-block:: bash

            source $HOME/pitzer_cpu_warpx.profile

      Finally, since pitzer does not yet provide software modules for some of our dependencies, install them once:

      .. code-block:: bash

         bash $HOME/src/warpx/Tools/machines/pitzer-osc/install_cpu_dependencies.sh

      .. dropdown:: Script Details
         :color: light
         :icon: info
         :animate: fade-in-slide-down

         .. literalinclude:: ../../../../Tools/machines/pitzer-osc/install_cpu_dependencies.sh
            :language: bash


.. _building-pitzer-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. tab-set::
   .. tab-item:: V100 GPUs

      .. code-block:: bash

         cd $HOME/src/warpx
         rm -rf build_v100

         cmake -S . -B build_v100 -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
         cmake --build build_v100 -j 48

      The WarpX application executables are now in ``$HOME/src/warpx/build_v100/bin/``. Additionally, the following commands will install WarpX as a Python module:

      .. code-block:: bash

         cd $HOME/src/warpx
         rm -rf build_v100_py

         cmake -S . -B build_v100_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
         cmake --build build_v100_py -j 48 --target pip_install

   .. tab-item:: CPU Nodes

      .. code-block:: bash

         cd $HOME/src/warpx
         rm -rf build

         cmake -S . -B build -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
         cmake --build build -j 48

      The WarpX application executables are now in ``$HOME/src/warpx/build/bin/``. Additionally, the following commands will install WarpX as a Python module:

      .. code-block:: bash

         cd $HOME/src/warpx
         rm -rf build_py

         cmake -S . -B build_py -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
         cmake --build build_py -j 48 --target pip_install

Now, you can :ref:`submit Pitzer compute jobs <running-pitzer>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`). Or, you can use the WarpX executables to submit Pitzer jobs (:ref:`example inputs <usage-examples>`). For executables, you can reference their location in your :ref:`job script <running-pitzer>` or copy them to a location in ``/scratch``.

.. _building-pitzer-update:

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

- :ref:`update the pitzer_cpu_warpx.profile file <building-pitzer-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-pitzer-preparation>`.

As a last step, clean the build directory ``rm -rf $HOME/src/warpx/build_*`` and rebuild WarpX.

.. _running-pitzer:

Running
-------

.. tab-set::

   .. tab-item:: V100 GPUs

      Pitzer's GPU partition includes:

      - 32 nodes, each equipped with two V100 (16GB) GPUs.
      - 42 nodes, each with two V100 (32GB) GPUs.
      - 4 large memory nodes, each with quad V100 (32GB) GPUs.

      To run a WarpX simulation on the GPU nodes, use the batch script provided below. Adjust the ``-N`` parameter in the script to match the number of nodes you intend to use. Each node in this partition supports running one MPI rank per GPU.

      .. literalinclude:: ../../../../Tools/machines/pitzer-osc/pitzer_v100.sbatch
         :language: bash
         :caption: Copy this file from ``$HOME/src/warpx/Tools/machines/pitzer-osc/pitzer_v100.sbatch``.

      After preparing your script, submit your job with the following command:

      .. code-block:: bash

         sbatch pitzer_v100.sbatch

   .. tab-item:: CPU Nodes

      For CPU-based computations, Pitzer offers:

      - 224 nodes, each with dual Intel Xeon Gold 6148 CPUs and 192 GB RAM.
      - 340 nodes, each with dual Intel Xeon Platinum 8268 CPUs and 192 GB RAM.
      - 16 large memory nodes.

      To submit a job to the CPU partition, use the provided batch script. Ensure you have copied the script to your working directory.

      .. literalinclude:: ../../../../Tools/machines/pitzer-osc/pitzer_cpu.sbatch
         :language: bash
         :caption: Copy this file from ``$HOME/src/warpx/Tools/machines/pitzer-osc/pitzer_cpu.sbatch``.

      Submit your job with:

      .. code-block:: bash

         sbatch pitzer_cpu.sbatch

.. _post-processing-osc:

Post-Processing
---------------

For post-processing, many users prefer to use the online `Jupyter service <https://ondemand.osc.edu/pun/sys/dashboard/batch_connect/sessions>`__ (`documentation <https://www.osc.edu/vocabulary/documentation/jupyter>`__) that is directly connected to the cluster's fast filesystem.

.. note::

   This section is a stub and contributions are welcome.
   We can document further details, e.g., which recommended post-processing Python software to install or how to customize Jupyter kernels here.
