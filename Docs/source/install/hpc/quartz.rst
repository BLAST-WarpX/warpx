.. _building-quartz:

Quartz (LLNL)
=============

The `Quartz Intel CPU cluster <https://hpc.llnl.gov/hardware/platforms/quartz>`_ is located at LLNL.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `LLNL user account <https://lc.llnl.gov/lorenz/mylc/mylc.cgi>`__ (login required)
* `Quartz user guide <https://computing.llnl.gov/tutorials/linux_clusters/>`_
* Batch system: `Slurm <https://computing.llnl.gov/tutorials/moab/>`_
* `Jupyter service <https://lc.llnl.gov/jupyter>`__ (`documentation <https://lc.llnl.gov/confluence/display/LC/JupyterHub+and+Jupyter+Notebook>`__, login required)
* `Production directories <https://hpc.llnl.gov/hardware/file-systems>`_:

  * ``/p/lustre1/$(whoami)`` and ``/p/lustre2/$(whoami)``: personal directory on the parallel filesystem
  * Note that the ``$HOME`` directory and the ``/usr/workspace/$(whoami)`` space are NFS mounted and *not* suitable for production quality data generation.


.. _building-quartz-preparation:

Preparation
-----------

Use the following commands to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx

We use system software modules, add environment hints and further dependencies via the file ``$HOME/quartz_warpx.profile``.
Create it now:

.. code-block:: bash

   cp $HOME/src/warpx/Tools/machines/quartz-llnl/quartz_warpx.profile.example $HOME/quartz_warpx.profile

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/quartz-llnl/quartz_warpx.profile.example
      :language: bash

Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
For example, if you are member of the project ``tps``, then run ``vi $HOME/quartz_warpx.profile``.
Enter the edit mode by typing ``i`` and edit line 2 to read:

.. code-block:: bash

   export proj="tps"

Exit the ``vi`` editor with ``Esc`` and then type ``:wq`` (write & quit).

.. important::

   Now, and as the first step on future logins to Quartz, activate these environment settings:

   .. code-block:: bash

      source $HOME/quartz_warpx.profile

Finally, since Quartz does not yet provide software modules for some of our dependencies, install them once:

.. code-block:: bash

   bash $HOME/src/warpx/Tools/machines/quartz-llnl/install_dependencies.sh
   source /usr/workspace/${USER}/quartz/venvs/warpx-quartz/bin/activate

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/quartz-llnl/install_dependencies.sh
      :language: bash


.. _building-quartz-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build_quartz

   cmake -S . -B build_quartz -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_quartz -j 6

The WarpX application executables are now in ``$HOME/src/warpx/build_quartz/bin/``.
Additionally, the following commands will install WarpX as a Python module:

.. code-block:: bash

   rm -rf build_quartz_py

   cmake -S . -B build_quartz_py -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_quartz_py -j 6 --target pip_install

Now, you can :ref:`submit Quartz compute jobs <running-cpp-quartz>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit Quartz jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-cpp-quartz>` or copy them to a location in ``$PROJWORK/$proj/``.


.. _building-quartz-update:

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
   git log     # press q to exit

And, if needed,

- :ref:`update the quartz_warpx.profile file <building-quartz-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-quartz-preparation>`.

As a last step, clean the build directory ``rm -rf $HOME/src/warpx/build_quartz`` and rebuild WarpX.


.. _running-cpp-quartz:

Running
-------

.. _running-cpp-quartz-CPUs:

Intel Xeon E5-2695 v4 CPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^

The batch script below can be used to run a WarpX simulation on 2 nodes on the supercomputer Quartz at LLNL.
Replace descriptions between chevrons ``<>`` by relevant values, for instance ``<input file>`` could be ``plasma_mirror_inputs``.

.. literalinclude:: ../../../../Tools/machines/quartz-llnl/quartz.sbatch
   :language: bash
   :caption: You can copy this file from ``Tools/machines/quartz-llnl/quartz.sbatch``.

To run a simulation, copy the lines above to a file ``quartz.sbatch`` and run

.. code-block:: bash

   sbatch quartz.sbatch

to submit the job.
