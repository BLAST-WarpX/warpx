.. _building-lassen:

Lassen (LLNL)
=============

The `Lassen V100 GPU cluster <https://hpc.llnl.gov/hardware/platforms/lassen>`__ is located at LLNL.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `LLNL user account <https://lc.llnl.gov/lorenz/mylc/mylc.cgi>`__ (login required)
* `Lassen user guide <https://hpc.llnl.gov/training/tutorials/using-lcs-sierra-system>`__
* Batch system: `LSF <https://hpc.llnl.gov/training/tutorials/using-lcs-sierra-system#batch-system>`__
* `Jupyter service <https://lc.llnl.gov/jupyter>`__ (`documentation <https://lc.llnl.gov/confluence/display/LC/JupyterHub+and+Jupyter+Notebook>`__, login required)
* `Production directories <https://hpc.llnl.gov/hardware/file-systems>`__:

  * ``/p/gpfs1/$(whoami)``: personal directory on the parallel filesystem
  * Note that the ``$HOME`` directory and the ``/usr/workspace/$(whoami)`` space are NFS mounted and *not* suitable for production quality data generation.


Login
-----

.. code-block:: bash

   ssh lassen.llnl.gov


.. _building-lassen-preparation:

Preparation
-----------

Use the following commands to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git /usr/workspace/${USER}/lassen/src/warpx

We use system software modules, add environment hints and further dependencies via the file ``$HOME/lassen_v100_warpx_toss3.profile``.
Create it now:

.. code-block:: bash

   cp /usr/workspace/${USER}/lassen/src/warpx/Tools/machines/lassen-llnl/lassen_v100_warpx_toss3.profile.example $HOME/lassen_v100_warpx_toss3.profile

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/lassen-llnl/lassen_v100_warpx_toss3.profile.example
      :language: bash

Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
For example, if you are member of the project ``nsldt``, then run ``vi $HOME/lassen_v100_warpx_toss3.profile``.
Enter the edit mode by typing ``i`` and edit line 2 to read:

.. code-block:: bash

   export proj="nsldt"

Exit the ``vi`` editor with ``Esc`` and then type ``:wq`` (write & quit).

.. important::

   Now, and as the first step on future logins to lassen, activate these environment settings:

   .. code-block:: bash

      source $HOME/lassen_v100_warpx_toss3.profile

Finally, since lassen does not yet provide software modules for some of our dependencies, install them once:

.. code-block:: bash

   bash /usr/workspace/${USER}/lassen/src/warpx/Tools/machines/lassen-llnl/install_v100_dependencies_toss3.sh
   source /usr/workspace/${USER}/lassen-toss3/gpu/venvs/warpx-lassen-toss3/bin/activate

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/lassen-llnl/install_v100_dependencies_toss3.sh
      :language: bash


.. _building-lassen-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. code-block:: bash

   cd /usr/workspace/${USER}/lassen/src/warpx
   rm -rf build_lassen

   cmake -S . -B build_lassen -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_lassen -j 8

The WarpX application executables are now in ``/usr/workspace/${USER}/lassen/src/warpx/build_lassen/bin/``.
Additionally, the following commands will install WarpX as a Python module:

.. code-block:: bash

   rm -rf build_lassen_py

   cmake -S . -B build_lassen_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_lassen_py -j 8 --target pip_install

Now, you can :ref:`submit lassen compute jobs <running-cpp-lassen>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit lassen jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-cpp-lassen>` or copy them to a location in ``$PROJWORK/$proj/``.


.. _building-lassen-update:

Update WarpX & Dependencies
---------------------------

If you already installed WarpX in the past and want to update it, start by getting the latest source code:

.. code-block:: bash

   cd /usr/workspace/${USER}/lassen/src/warpx

   # read the output of this command - does it look ok?
   git status

   # get the latest WarpX source code
   git fetch
   git pull

   # read the output of these commands - do they look ok?
   git status
   git log     # press q to exit

And, if needed,

- :ref:`update the lassen_v100_warpx.profile file <building-lassen-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-lassen-preparation>`.

As a last step, clean the build directory ``rm -rf /usr/workspace/${USER}/lassen/src/warpx/build_lassen`` and rebuild WarpX.


.. _running-cpp-lassen:

Running
-------

.. _running-cpp-lassen-V100-GPUs:

V100 GPUs (16GB)
^^^^^^^^^^^^^^^^

The batch script below can be used to run a WarpX simulation on 2 nodes on the supercomputer Lassen at LLNL.
Replace descriptions between chevrons ``<>`` by relevant values, for instance ``<input file>`` could be ``plasma_mirror_inputs``.
Note that the only option so far is to run with one MPI rank per GPU.

.. literalinclude:: ../../../../Tools/machines/lassen-llnl/lassen_v100.bsub
   :language: bash
   :caption: You can copy this file from ``Tools/machines/lassen-llnl/lassen_v100.bsub``.

To run a simulation, copy the lines above to a file ``lassen_v100.bsub`` and run

.. code-block:: bash

   bsub lassen_v100.bsub

to submit the job.

For a 3D simulation with a few (1-4) particles per cell using FDTD Maxwell
solver on V100 GPUs for a well load-balanced problem (in our case laser
wakefield acceleration simulation in a boosted frame in the quasi-linear
regime), the following set of parameters provided good performance:

* ``amr.max_grid_size=256`` and ``amr.blocking_factor=128``.

* **One MPI rank per GPU** (e.g., 4 MPI ranks for the 4 GPUs on each Lassen
  node)

* **Two `128x128x128` grids per GPU**, or **one `128x128x256` grid per GPU**.


.. _building-lassen-issues:

Known System Issues
-------------------

.. warning::

   Feb 17th, 2022 (INC0278922):
   The implementation of ``AllGatherv`` in IBM's MPI optimization library "libcollectives" is broken and leads to HDF5 crashes for multi-node runs.

   Our batch script templates above `apply this work-around <https://github.com/ECP-WarpX/WarpX/pull/2874>`__ *before* the call to ``jsrun``, which avoids the broken routines from IBM and trades them for an OpenMPI implementation of collectives:

   .. code-block:: bash

      export OMPI_MCA_coll_ibm_skip_allgatherv=true

As part of the same `CORAL acquisition program <https://doi.org/10.1109/SC.2018.00055>`__, Lassen is very similar to the design of Summit (OLCF).
Thus, when encountering new issues it is worth checking also the :ref:`known Summit issues and work-arounds <building-summit-issues>`.
