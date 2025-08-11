.. _building-aurora:

Aurora (ALCF)
==============

The `Aurora cluster <https://docs.alcf.anl.gov/aurora/>`_ is located at ALCF.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `ALCF user guide <https://docs.alcf.anl.gov/>`__
* Batch system: `PBS <https://docs.alcf.anl.gov/running-jobs/>`__
* `Filesystems <https://docs.alcf.anl.gov/data-management/filesystem-and-storage/>`__:

  * ``/lus/flare/projects/$proj/``: shared with all members of a project, Lustre

.. _building-aurora-preparation:

Preparation
-----------

Use the following commands to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/BLAST-WarpX/warpx.git $HOME/src/warpx

We use system software modules, add environment hints and further dependencies via the file ``$HOME/aurora_warpx.profile``.
Create it now:

.. code-block:: bash

   cp $HOME/src/warpx/Tools/machines/aurora-alcf/aurora_warpx.profile.example $HOME/aurora_warpx.profile

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/aurora-alcf/aurora_warpx.profile.example
      :language: bash

Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
For example, if you are member of the project ``proj_name``, then run ``nano $HOME/aurora_warpx.profile`` and edit line 2 to read:

.. code-block:: bash

   export proj="proj_name"

Exit the ``nano`` editor with ``Ctrl`` + ``O`` (save) and then ``Ctrl`` + ``X`` (exit).

.. important::

   Now, and as the first step on future logins to Aurora, activate these environment settings:

   .. code-block:: bash

      source $HOME/aurora_warpx.profile

Finally, since Aurora does not yet provide software modules for some of our dependencies, install them once:

.. code-block:: bash

   bash $HOME/src/warpx/Tools/machines/aurora-alcf/install_dependencies.sh
   source ${CFS}/${proj%_g}/${USER}/sw/aurora/gpu/venvs/warpx/bin/activate

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/aurora-alcf/install_dependencies.sh
      :language: bash


.. _building-aurora-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build_aurora

   cmake -S . -B build_aurora -DWarpX_COMPUTE=SYCL -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_aurora -j 16

The WarpX application executables are now in ``$HOME/src/warpx/build_aurora/bin/``.
Additionally, the following commands will install WarpX as a Python module:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build_aurora_py

   cmake -S . -B build_aurora_py -DWarpX_COMPUTE=SYCL -DWarpX_FFT=OFF -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_aurora_py -j 16 --target pip_install

Now, you can :ref:`submit Aurora compute jobs <running-cpp-aurora>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit Aurora jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-cpp-aurora>` or copy them to a location in ``/lus/flare/projects/$proj/``.


.. _building-aurora-update:

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

- :ref:`update the aurora_warpx.profile or aurora_cpu_warpx files <building-aurora-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-aurora-preparation>`.

As a last step, clean the build directory ``rm -rf $HOME/src/warpx/build_aurora*`` and rebuild WarpX.


.. _running-cpp-aurora:

Running
-------

The batch script below can be used to run a WarpX simulation on multiple nodes (change ``<NODES>`` accordingly) on the supercomputer Aurora at ALCF.

Replace descriptions between chevrons ``<>`` by relevant values, for instance ``<input file>`` could be ``plasma_mirror_inputs``.
Note that we run one MPI rank per GPU.

.. literalinclude:: ../../../../Tools/machines/aurora-alcf/aurora.pbs
   :language: bash
   :caption: You can copy this file from ``$HOME/src/warpx/Tools/machines/aurora-alcf/aurora.pbs``.

To run a simulation, copy the lines above to a file ``aurora.pbs`` and run

.. code-block:: bash

   qsub aurora.pbs

to submit the job.
