.. _building-greatlakes:

Great Lakes (UMich)
===================

The `Great Lakes cluster <https://arc.umich.edu/greatlakes/>`_ is located at University of Michigan.
The cluster has various partitions, including `GPU nodes and CPU nodes <https://arc.umich.edu/greatlakes/configuration/>`__.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `Great Lakes user guide <https://arc.umich.edu/greatlakes/>`__
* Batch system: `Slurm <https://arc.umich.edu/greatlakes/slurm-user-guide/>`__
* `Jupyter service <https://greatlakes.arc-ts.umich.edu>`__ (`documentation <https://arc.umich.edu/greatlakes/user-guide/#document-2>`__)
* `Filesystems <https://arc.umich.edu/greatlakes/user-guide/#document-1>`__:

  * ``$HOME``: per-user directory, use only for inputs, source and scripts; backed up (80GB)
  * ``/scratch``: per-project `production directory <https://arc.umich.edu/greatlakes/user-guide/#scratchpolicies>`__; very fast for parallel jobs; purged every 60 days (10TB default)


.. _building-greatlakes-preparation:

Preparation
-----------

Use the following commands to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx

On Great Lakes, you can run either on GPU nodes with `fast V100 GPUs (recommended), the even faster A100 GPUs (only a few available) or CPU nodes <https://arc.umich.edu/greatlakes/configuration/>`__.

.. tab-set::

   .. tab-item:: V100 GPUs

      We use system software modules, add environment hints and further dependencies via the file ``$HOME/greatlakes_v100_warpx.profile``.
      Create it now:

      .. code-block:: bash

         cp $HOME/src/warpx/Tools/machines/greatlakes-umich/greatlakes_v100_warpx.profile.example $HOME/greatlakes_v100_warpx.profile

      .. dropdown:: Script Details
         :color: light
         :icon: info
         :animate: fade-in-slide-down

         .. literalinclude:: ../../../../Tools/machines/greatlakes-umich/greatlakes_v100_warpx.profile.example
            :language: bash

      Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
      For example, if you are member of the project ``iloveplasma``, then run ``nano $HOME/greatlakes_v100_warpx.profile`` and edit line 2 to read:

      .. code-block:: bash

         export proj="iloveplasma"

      Exit the ``nano`` editor with ``Ctrl`` + ``O`` (save) and then ``Ctrl`` + ``X`` (exit).

      .. important::

         Now, and as the first step on future logins to Great Lakes, activate these environment settings:

         .. code-block:: bash

            source $HOME/greatlakes_v100_warpx.profile

      Finally, since Great Lakes does not yet provide software modules for some of our dependencies, install them once:

      .. code-block:: bash

         bash $HOME/src/warpx/Tools/machines/greatlakes-umich/install_v100_dependencies.sh
         source ${HOME}/sw/greatlakes/v100/venvs/warpx-v100/bin/activate

      .. dropdown:: Script Details
         :color: light
         :icon: info
         :animate: fade-in-slide-down

         .. literalinclude:: ../../../../Tools/machines/greatlakes-umich/install_v100_dependencies.sh
            :language: bash


   .. tab-item:: A100 Nodes

      .. note::

         This section is TODO.


   .. tab-item:: CPU Nodes

      .. note::

         This section is TODO.


.. _building-greatlakes-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. tab-set::

   .. tab-item:: V100 GPUs

      .. code-block:: bash

         cd $HOME/src/warpx
         rm -rf build_v100

         cmake -S . -B build_v100 -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
         cmake --build build_v100 -j 8

      The WarpX application executables are now in ``$HOME/src/warpx/build_v100/bin/``.
      Additionally, the following commands will install WarpX as a Python module:

      .. code-block:: bash

         cd $HOME/src/warpx
         rm -rf build_v100_py

         cmake -S . -B build_v100_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
         cmake --build build_v100_py -j 8 --target pip_install


   .. tab-item:: A100 Nodes

      .. note::

         This section is TODO.


   .. tab-item:: CPU Nodes

      .. note::

         This section is TODO.

Now, you can :ref:`submit Great Lakes compute jobs <running-cpp-greatlakes>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit greatlakes jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-cpp-greatlakes>` or copy them to a location in ``/scratch``.


.. _building-greatlakes-update:

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

- :ref:`update the greatlakes_v100_warpx.profile file <building-greatlakes-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-greatlakes-preparation>`.

As a last step, clean the build directory ``rm -rf $HOME/src/warpx/build_*`` and rebuild WarpX.


.. _running-cpp-greatlakes:

Running
-------

.. tab-set::

   .. tab-item:: V100 (16GB) GPUs

      The batch script below can be used to run a WarpX simulation on multiple nodes (change ``-N`` accordingly) on the supercomputer Great Lakes at University of Michigan.
      This partition has `20 nodes, each with two V100 GPUs <https://arc.umich.edu/greatlakes/configuration/>`__.

      Replace descriptions between chevrons ``<>`` by relevant values, for instance ``<input file>`` could be ``plasma_mirror_inputs``.
      Note that we run one MPI rank per GPU.

      .. literalinclude:: ../../../../Tools/machines/greatlakes-umich/greatlakes_v100.sbatch
         :language: bash
         :caption: You can copy this file from ``$HOME/src/warpx/Tools/machines/greatlakes-umich/greatlakes_v100.sbatch``.

      To run a simulation, copy the lines above to a file ``greatlakes_v100.sbatch`` and run

      .. code-block:: bash

         sbatch greatlakes_v100.sbatch

      to submit the job.


   .. tab-item:: A100 (80GB) GPUs

      This partition has `2 nodes, each with four A100 GPUs <https://arc.umich.edu/greatlakes/configuration/>`__ that provide 80 GB HBM per A100 GPU.
      To the user, each node will appear as if it has 8 A100 GPUs with 40 GB memory each.

      .. note::

         This section is TODO.


   .. tab-item:: CPU Nodes

      The Great Lakes CPU partition as up to `455 nodes <https://arc.umich.edu/greatlakes/configuration/>`__, each with 2x Intel Xeon Gold 6154 CPUs and 180 GB RAM.

      .. note::

         This section is TODO.


.. _post-processing-greatlakes:

Post-Processing
---------------

For post-processing, many users prefer to use the online `Jupyter service <https://greatlakes.arc-ts.umich.edu>`__ (`documentation <https://arc.umich.edu/greatlakes/user-guide/#document-2>`__) that is directly connected to the cluster's fast filesystem.

.. note::

   This section is a stub and contributions are welcome.
   We can document further details, e.g., which recommended post-processing Python software to install or how to customize Jupyter kernels here.
