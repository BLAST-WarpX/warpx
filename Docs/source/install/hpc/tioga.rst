.. _building-tioga:

Tioga (LLNL)
============

The `Tioga AMD GPU cluster <https://hpc.llnl.gov/hardware/compute-platforms/tioga>`__ is located at LLNL.
It is equipped with two nodes that have each four AMD MI300A APUs.
Tioga is an LLNL El Capitan Early Access System.

There are also "conventional" MI250X GPUs on Tioga nodes, which we did not yet document.
El Capitan will use MI300A GPUs.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `LLNL user account <https://lc.llnl.gov/lorenz/mylc/mylc.cgi>`__ (login required)
* `Tioga user guide <https://lc.llnl.gov/confluence/x/7A_sK>`__
* Batch system: `Flux with Slurm Wrappers <https://lc.llnl.gov/confluence/display/ELCAPEA/Running+Jobs>`__
* `Jupyter service <https://lc.llnl.gov/jupyter>`__ (`documentation <https://lc.llnl.gov/confluence/display/LC/JupyterHub+and+Jupyter+Notebook>`__, login required)
* `Production directories <https://lc.llnl.gov/confluence/display/ELCAPEA/File+Systems>`__:

  * ``/p/lustre1/${USER}``: personal directory on the parallel filesystem (also: ``lustre2``)
  * Note that the ``$HOME`` directory and the ``/usr/workspace/${USER}`` space are NFS mounted and *not* suitable for production quality data generation.


Login
-----

.. code-block:: bash

   ssh tioga.llnl.gov

To use the available MI300A nodes (currently two), request one via

.. code-block:: bash

   salloc -N 1 -p mi300a -t 30:0


.. _building-tioga-preparation:

Preparation
-----------

Use the following commands to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git /p/lustre1/${USER}/tioga/src/warpx

We use system software modules, add environment hints and further dependencies via the file ``$HOME/tioga_mi300a_warpx.profile``.
Create it now:

.. code-block:: bash

   cp /p/lustre1/${USER}/tioga/src/warpx/Tools/machines/tioga-llnl/tioga_mi300a_warpx.profile.example $HOME/tioga_mi300a_warpx.profile

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/tioga-llnl/tioga_mi300a_warpx.profile.example
      :language: bash

Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
**Currently, this is unused and can be kept empty.**
Once project allocation becomes required, e.g., if you are member of the project ``abcde``, then run ``vi $HOME/tioga_mi300a_warpx.profile``.
Enter the edit mode by typing ``i`` and edit line 2 to read:

.. code-block:: bash

   export proj="abcde"

Exit the ``vi`` editor with ``Esc`` and then type ``:wq`` (write & quit).

.. important::

   Now, and as the first step on future logins to Tioga, activate these environment settings:

   .. code-block:: bash

      source $HOME/tioga_mi300a_warpx.profile

Finally, since Tioga does not yet provide software modules for some of our dependencies, install them once:


  .. code-block:: bash

     bash /p/lustre1/${USER}/tioga/src/warpx/Tools/machines/tioga-llnl/install_mi300a_dependencies.sh
     source /p/lustre1/${USER}/tioga/warpx/mi300a/gpu/venvs/warpx-trioga-mi300a/bin/activate

  .. dropdown:: Script Details
     :color: light
     :icon: info
     :animate: fade-in-slide-down

     .. literalinclude:: ../../../../Tools/machines/tioga-llnl/install_mi300a_dependencies.sh
        :language: bash

  .. dropdown:: AI/ML Dependencies (Optional)
     :animate: fade-in-slide-down

     If you plan to run AI/ML workflows depending on PyTorch et al., run the next step as well.
     This will take a while and should be skipped if not needed.

     .. code-block:: bash

        bash /p/lustre1/${USER}/tioga/src/warpx/Tools/machines/tioga-llnl/install_mi300a_ml.sh

     .. dropdown:: Script Details
        :color: light
        :icon: info
        :animate: fade-in-slide-down

        .. literalinclude:: ../../../../Tools/machines/tioga-llnl/install_mi300a_ml.sh
           :language: bash


.. _building-tioga-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. code-block:: bash

   cd /p/lustre1/${USER}/tioga/src/warpx

   cmake --fresh -S . -B build_tioga -DWarpX_COMPUTE=HIP -DWarpX_FFT=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_tioga -j 24

The WarpX application executables are now in ``/p/lustre1/${USER}/tioga/src/warpx/build_tioga/bin/``.
Additionally, the following commands will install WarpX as a Python module:

.. code-block:: bash

   cmake --fresh -S . -B build_tioga_py -DWarpX_COMPUTE=HIP -DWarpX_FFT=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_tioga_py -j 24 --target pip_install

Now, you can :ref:`submit tioga compute jobs <running-cpp-tioga>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit tioga jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-cpp-tioga>` or copy them to a location in ``$PROJWORK/$proj/``.


.. _building-tioga-update:

Update WarpX & Dependencies
---------------------------

If you already installed WarpX in the past and want to update it, start by getting the latest source code:

.. code-block:: bash

   cd /p/lustre1/${USER}/tioga/src/warpx

   # read the output of this command - does it look ok?
   git status

   # get the latest WarpX source code
   git fetch
   git pull

   # read the output of these commands - do they look ok?
   git status
   git log     # press q to exit

And, if needed,

- :ref:`update the tioga_mi300a_warpx.profile file <building-tioga-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-tioga-preparation>`.

As a last step :ref:`rebuild WarpX <building-tioga-compilation>`.


.. _running-cpp-tioga:

Running
-------

.. _running-cpp-tioga-MI300A-APUs:

MI300A APUs (128GB)
^^^^^^^^^^^^^^^^^^^

The batch script below can be used to run a WarpX simulation on 1 node with 4 APUs on the supercomputer Tioga at LLNL.
Replace descriptions between chevrons ``<>`` by relevant values, for instance ``<input file>`` could be ``plasma_mirror_inputs``.
WarpX runs with one MPI rank per GPU.

Note that we append these non-default runtime options:

* ``amrex.use_gpu_aware_mpi=1``: make use of fast APU to APU MPI communications
* ``amrex.the_arena_init_size=1``: avoid overallocating memory that is *shared* on APUs between CPU & GPU

.. literalinclude:: ../../../../Tools/machines/tioga-llnl/tioga_mi300a.sbatch
   :language: bash
   :caption: You can copy this file from ``Tools/machines/tioga-llnl/tioga_mi300a.sbatch``.

To run a simulation, copy the lines above to a file ``tioga_mi300a.sbatch`` and run

.. code-block:: bash

   sbatch tioga_mi300a.sbatch

to submit the job.
