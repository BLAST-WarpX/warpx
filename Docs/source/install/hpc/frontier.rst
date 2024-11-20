.. _building-frontier:

Frontier (OLCF)
===============

The `Frontier cluster <https://www.olcf.ornl.gov/frontier/>`_ is located at OLCF.

On Frontier, each compute node provides four AMD MI250X GPUs, each with two Graphics Compute Dies (GCDs) for a total of 8 GCDs per node.
You can think of the 8 GCDs as 8 separate GPUs, each having 64 GB of high-bandwidth memory (HBM2E).


Introduction
------------

If you are new to this system, **please see the following resources**:

* `Frontier user guide <https://docs.olcf.ornl.gov/systems/frontier_user_guide.html>`_
* Batch system: `Slurm <https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#running-jobs>`_
* `Filesystems <https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#data-and-storage>`_:

  * ``$HOME``: per-user directory, use only for inputs, source and scripts; backed up; mounted as read-only on compute nodes, that means you cannot run in it (50 GB quota)
  * ``$PROJWORK/$proj/``: shared with all members of a project, purged every 90 days, Lustre (recommended)
  * ``$MEMBERWORK/$proj/``: single user, purged every 90 days, Lustre (usually smaller quota, 50TB default quota)
  * ``$WORLDWORK/$proj/``: shared with all users, purged every 90 days, Lustre (50TB default quota)

Note: the Orion Lustre filesystem on Frontier and the older Alpine GPFS filesystem on Summit are not mounted on each others machines.
Use `Globus <https://www.globus.org>`__ to transfer data between them if needed.


.. _building-frontier-preparation:

Preparation
-----------

Use the following commands to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx

We use system software modules, add environment hints and further dependencies via the file ``$HOME/frontier_warpx.profile``.
Create it now:

.. code-block:: bash

   cp $HOME/src/warpx/Tools/machines/frontier-olcf/frontier_warpx.profile.example $HOME/frontier_warpx.profile

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/frontier-olcf/frontier_warpx.profile.example
      :language: bash

Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
For example, if you are member of the project ``aph114``, then run ``vi $HOME/frontier_warpx.profile``.
Enter the edit mode by typing ``i`` and edit line 2 to read:

.. code-block:: bash

   export proj="aph114"

Exit the ``vi`` editor with ``Esc`` and then type ``:wq`` (write & quit).

.. important::

   Now, and as the first step on future logins to Frontier, activate these environment settings:

   .. code-block:: bash

      source $HOME/frontier_warpx.profile

Finally, since Frontier does not yet provide software modules for some of our dependencies, install them once:

.. code-block:: bash

   bash $HOME/src/warpx/Tools/machines/frontier-olcf/install_dependencies.sh
   source $HOME/sw/frontier/gpu/venvs/warpx-frontier/bin/activate

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/frontier-olcf/install_dependencies.sh
      :language: bash


.. _building-frontier-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build_frontier

   cmake -S . -B build_frontier -DWarpX_COMPUTE=HIP -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_frontier -j 16

The WarpX application executables are now in ``$HOME/src/warpx/build_frontier/bin/``.
Additionally, the following commands will install WarpX as a Python module:

.. code-block:: bash

   rm -rf build_frontier_py

   cmake -S . -B build_frontier_py -DWarpX_COMPUTE=HIP -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_frontier_py -j 16 --target pip_install

Now, you can :ref:`submit Frontier compute jobs <running-cpp-frontier>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit Frontier jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-cpp-frontier>` or copy them to a location in ``$PROJWORK/$proj/``.


.. _building-frontier-update:

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

- :ref:`update the frontier_warpx.profile file <building-frontier-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-frontier-preparation>`.

As a last step, clean the build directory ``rm -rf $HOME/src/warpx/build_frontier`` and rebuild WarpX.


.. _running-cpp-frontier:

Running
-------

.. _running-cpp-frontier-MI250X-GPUs:

MI250X GPUs (2x64 GB)
^^^^^^^^^^^^^^^^^^^^^

After requesting an interactive node with the ``getNode`` alias above, run a simulation like this, here using 8 MPI ranks and a single node:

.. code-block:: bash

   runNode ./warpx inputs

Or in non-interactive runs:

.. literalinclude:: ../../../../Tools/machines/frontier-olcf/submit.sh
   :language: bash
   :caption: You can copy this file from ``$HOME/src/warpx/Tools/machines/frontier-olcf/submit.sh``.


.. _post-processing-frontier:

Post-Processing
---------------

For post-processing, most users use Python via OLCFs's `Jupyter service <https://jupyter.olcf.ornl.gov>`__ (`Docs <https://docs.olcf.ornl.gov/services_and_applications/jupyter/index.html>`__).

Please follow the same guidance as for :ref:`OLCF Summit post-processing <post-processing-summit>`.

.. _known-frontier-issues:

Known System Issues
-------------------

.. warning::

   May 16th, 2022 (OLCFHELP-6888):
   There is a caching bug in Libfabric that causes WarpX simulations to occasionally hang on Frontier on more than 1 node.

   As a work-around, please export the following environment variable in your job scripts until the issue is fixed:

   .. code-block:: bash

      #export FI_MR_CACHE_MAX_COUNT=0  # libfabric disable caching
      # or, less invasive:
      export FI_MR_CACHE_MONITOR=memhooks  # alternative cache monitor

.. warning::

   Sep 2nd, 2022 (OLCFDEV-1079):
   rocFFT in ROCm 5.1-5.3 tries to `write to a cache <https://rocfft.readthedocs.io/en/latest/#runtime-compilation>`__ in the home area by default.
   This does not scale, disable it via:

   .. code-block:: bash

      export ROCFFT_RTC_CACHE_PATH=/dev/null

.. warning::

   January, 2023 (OLCFDEV-1284, AMD Ticket: ORNLA-130):
   We discovered a regression in AMD ROCm, leading to 2x slower current deposition (and other slowdowns) in ROCm 5.3 and 5.4.

   June, 2023:
   Although a fix was planned for ROCm 5.5, we still see the same issue in this release and continue to exchange with AMD and HPE on the issue.

   Stay with the ROCm 5.2 module to avoid a 2x slowdown.

.. warning::

   August, 2023 (OLCFDEV-1597, OLCFHELP-12850, OLCFHELP-14253):
   With runs above 500 nodes, we observed issues in ``MPI_Waitall`` calls of the kind ``OFI Poll Failed UNDELIVERABLE``.
   According to the system known issues entry `OLCFDEV-1597 <https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#olcfdev-1597-ofi-poll-failed-undeliverable-errors>`__, we work around this by setting this environment variable in job scripts:

   .. code-block:: bash

      export MPICH_SMP_SINGLE_COPY_MODE=NONE
      export FI_CXI_RX_MATCH_MODE=software

.. warning::

   Checkpoints and I/O at scale seem to be slow with the default Lustre filesystem configuration.
   Please test checkpointing and I/O with short ``#SBATCH -q debug`` runs before running the full simulation.
   Execute ``lfs getstripe -d <dir>`` to show the default progressive file layout.
   Consider using ``lfs setstripe`` to change the `striping <https://wiki.lustre.org/Configuring_Lustre_File_Striping>`__ for new files **before** you submit the run.

   .. code-block:: bash

      mkdir /lustre/orion/proj-shared/<your-project>/<path/to/new/sim/dir>
      cd <new/sim/dir/above>
      # create your diagnostics directory first
      mkdir diags
      # change striping for new files before you submit the simulation
      #   this is an example, striping 10 MB blocks onto 32 nodes
      lfs setstripe -S 10M -c 32 diags

   Additionally, other AMReX users reported good performance for plotfile checkpoint/restart when using

   .. code-block:: ini

      amr.plot_nfiles = -1
      amr.checkpoint_nfiles = -1
      amrex.async_out_nfiles = 4096  # set to number of GPUs used
