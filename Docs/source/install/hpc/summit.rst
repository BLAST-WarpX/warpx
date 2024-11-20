.. _building-summit:

Summit (OLCF)
=============

The `Summit cluster <https://www.olcf.ornl.gov/summit/>`_ is located at OLCF.

On Summit, each compute node provides six V100 GPUs (16GB) and two Power9 CPUs.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `Summit user guide <https://docs.olcf.ornl.gov/systems/summit_user_guide.html>`_
* Batch system: `LSF <https://docs.olcf.ornl.gov/systems/summit_user_guide.html#running-jobs>`_
* `Jupyter service <https://jupyter.olcf.ornl.gov>`__
* `Filesystems <https://docs.olcf.ornl.gov/data/index.html#data-storage-and-transfers>`_:

  * ``$HOME``: per-user directory, use only for inputs, source and scripts; backed up; mounted as read-only on compute nodes, that means you cannot run in it (50 GB quota)
  * ``$PROJWORK/$proj/``: shared with all members of a project, purged every 90 days, GPFS (recommended)
  * ``$MEMBERWORK/$proj/``: single user, purged every 90 days, GPFS (usually smaller quota)
  * ``$WORLDWORK/$proj/``: shared with all users, purged every 90 days, GPFS
  * ``/ccs/proj/$proj/``: another, non-GPFS, file system for software and smaller data.

Note: the Alpine GPFS filesystem on Summit and the new Orion Lustre filesystem on Frontier are not mounted on each others machines.
Use `Globus <https://www.globus.org>`__ to transfer data between them if needed.


.. _building-summit-preparation:

Preparation
-----------

Use the following commands to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx

We use system software modules, add environment hints and further dependencies via the file ``$HOME/summit_warpx.profile``.
Create it now:

.. code-block:: bash

   cp $HOME/src/warpx/Tools/machines/summit-olcf/summit_warpx.profile.example $HOME/summit_warpx.profile

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/summit-olcf/summit_warpx.profile.example
      :language: bash

Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
For example, if you are member of the project ``aph114``, then run ``vi $HOME/summit_warpx.profile``.
Enter the edit mode by typing ``i`` and edit line 2 to read:

.. code-block:: bash

   export proj="aph114"

Exit the ``vi`` editor with ``Esc`` and then type ``:wq`` (write & quit).

.. important::

   Now, and as the first step on future logins to Summit, activate these environment settings:

   .. code-block:: bash

      source $HOME/summit_warpx.profile

Finally, since Summit does not yet provide software modules for some of our dependencies, install them once:

.. code-block:: bash

   bash $HOME/src/warpx/Tools/machines/summit-olcf/install_gpu_dependencies.sh
   source /ccs/proj/$proj/${USER}/sw/summit/gpu/venvs/warpx-summit/bin/activate

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/summit-olcf/install_gpu_dependencies.sh
      :language: bash

.. dropdown:: AI/ML Dependencies (Optional)
   :animate: fade-in-slide-down

   If you plan to run AI/ML workflows depending on pyTorch, run the next step as well.
   This will take a while and should be skipped if not needed.

   .. code-block:: bash

      runNode bash $HOME/src/warpx/Tools/machines/summit-olcf/install_gpu_ml.sh

   .. dropdown:: Script Details
      :color: light
      :icon: info
      :animate: fade-in-slide-down

      .. literalinclude:: ../../../../Tools/machines/summit-olcf/install_gpu_ml.sh
         :language: bash

   For `optimas dependencies <https://github.com/optimas-org/optimas>`__ (incl. scikit-learn), plan another hour of build time:

   .. code-block:: bash

      python3 -m pip install -r $HOME/src/warpx/Tools/optimas/requirements.txt


.. _building-summit-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build_summit

   cmake -S . -B build_summit -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_summit -j 8

The WarpX application executables are now in ``$HOME/src/warpx/build_summit/bin/``.
Additionally, the following commands will install WarpX as a Python module:

.. code-block:: bash

   rm -rf build_summit_py

   cmake -S . -B build_summit_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_summit_py -j 8 --target pip_install

Now, you can :ref:`submit Summit compute jobs <running-cpp-summit>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit Summit jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-cpp-summit>` or copy them to a location in ``$PROJWORK/$proj/``.


.. _building-summit-update:

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

- :ref:`update the summit_warpx.profile file <building-summit-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-summit-preparation>`.

As a last step, clean the build directory ``rm -rf $HOME/src/warpx/build_summit`` and rebuild WarpX.


.. _running-cpp-summit:

Running
-------

.. _running-cpp-summit-V100-GPUs:

V100 GPUs (16GB)
^^^^^^^^^^^^^^^^

The batch script below can be used to run a WarpX simulation on 2 nodes on
the supercomputer Summit at OLCF. Replace descriptions between chevrons ``<>``
by relevant values, for instance ``<input file>`` could be
``plasma_mirror_inputs``.
Note that WarpX runs with one MPI rank per GPU and there are 6 GPUs per node:

.. literalinclude:: ../../../../Tools/machines/summit-olcf/summit_v100.bsub
   :language: bash
   :caption: You can copy this file from ``Tools/machines/summit-olcf/summit_v100.bsub``.

To run a simulation, copy the lines above to a file ``summit_v100.bsub`` and
run
::

  bsub summit_v100.bsub

to submit the job.

For a 3D simulation with a few (1-4) particles per cell using FDTD Maxwell
solver on Summit for a well load-balanced problem (in our case laser
wakefield acceleration simulation in a boosted frame in the quasi-linear
regime), the following set of parameters provided good performance:

* ``amr.max_grid_size=256`` and ``amr.blocking_factor=128``.

* **One MPI rank per GPU** (e.g., 6 MPI ranks for the 6 GPUs on each Summit
  node)

* **Two `128x128x128` grids per GPU**, or **one `128x128x256` grid per GPU**.

A batch script with more options regarding profiling on Summit can be found at
:download:`Summit batch script <../../../../Tools/machines/summit-olcf/summit_profiling.bsub>`

.. _running-cpp-summit-Power9-CPUs:

Power9 CPUs
^^^^^^^^^^^

Similar to above, the batch script below can be used to run a WarpX simulation on
1 node on the supercomputer Summit at OLCF, on Power9 CPUs (i.e., the GPUs are
ignored).

.. literalinclude:: ../../../../Tools/machines/summit-olcf/summit_power9.bsub
   :language: bash
   :caption: You can copy this file from ``Tools/machines/summit-olcf/summit_power9.bsub``.

For a 3D simulation with a few (1-4) particles per cell using FDTD Maxwell
solver on Summit for a well load-balanced problem, the following set of
parameters provided good performance:

* ``amr.max_grid_size=64`` and ``amr.blocking_factor=64``

* **Two MPI ranks per node** (i.e. 2 resource sets per node; equivalently, 1
  resource set per socket)

* **21 physical CPU cores per MPI rank**

* **21 OpenMP threads per MPI rank** (i.e. 1 OpenMP thread per physical core)

* **SMT 1 (Simultaneous Multithreading level 1)**

* **Sixteen `64x64x64` grids per MPI rank** (with default tiling in WarpX, this
  results in ~49 tiles per OpenMP thread)

.. _building-summit-io-performance:

I/O Performance Tuning
----------------------

.. _building-summit-large-blocks:

GPFS Large Block I/O
^^^^^^^^^^^^^^^^^^^^

Setting ``IBM_largeblock_io`` to ``true`` disables data shipping, saving overhead when writing/reading large contiguous I/O chunks.

.. code-block:: bash

   export IBM_largeblock_io=true

.. _building-summit-romio-hints:

ROMIO MPI-IO Hints
^^^^^^^^^^^^^^^^^^

You might notice some parallel HDF5 performance improvements on Summit by setting the appropriate ROMIO hints for MPI-IO operations.

.. code-block:: bash

   export OMPI_MCA_io=romio321
   export ROMIO_HINTS=./romio-hints

You can generate the ``romio-hints`` by issuing the following command. Remember to change the number of ``cb_nodes`` to match the number of compute nodes you are using (example here: ``64``).

.. code-block:: bash

   cat > romio-hints << EOL
   romio_cb_write enable
   romio_ds_write enable
   cb_buffer_size 16777216
   cb_nodes 64
   EOL

The ``romio-hints`` file contains pairs of key-value hints to enable and tune collective
buffering of MPI-IO operations. As Summit's Alpine file system uses a 16MB block size,
you should set the collective buffer size to 16GB and tune the number of aggregators
(``cb_nodes``) to the number of compute nodes you are using, i.e., one aggregator per node.

Further details are available at `Summit's documentation page <https://docs.olcf.ornl.gov/systems/summit_user_guide.html#slow-performance-using-parallel-hdf5-resolved-march-12-2019>`__.

.. _building-summit-issues:

Known System Issues
-------------------

.. warning::

   Sep 16th, 2021 (OLCFHELP-3685):
   The **Jupyter** service cannot open HDF5 files without hanging, due to a filesystem mounting problem.

   `Please apply this work-around <https://github.com/openPMD/openPMD-api/pull/1106>`__ in a Jupyter cell before opening any HDF5 files for read:

   .. code-block:: python3

      import os
      os.environ['HDF5_USE_FILE_LOCKING'] = "FALSE"

.. warning::

   Aug 27th, 2021 (OLCFHELP-3442):
   Created simulation files and directories are no longer accessible by your team members, even if you create them on ``$PROJWORK``.
   Setting the proper "user mask" (``umask``) does not yet work to fix this.

   Please run those commands *after* running a simulation to fix this.
   You can also append this to the end of your job scripts *after* the ``jsrun`` line:

   .. code-block:: bash

      # cd your-simulation-directory
      find . -type d -exec chmod g+rwx {} \;
      find . -type f -exec chmod g+rw {} \;

.. warning::

   Sep 3rd, 2021 (OLCFHELP-3545):
   The implementation of barriers in IBM's MPI fork is broken and leads to crashes at scale.
   This is seen with runs using 200 nodes and above.

   Our batch script templates above `apply this work-around <https://github.com/ECP-WarpX/WarpX/pull/2283>`__ *before* the call to ``jsrun``, which avoids the broken routines from IBM and trades them for an OpenMPI implementation of collectives:

   .. code-block:: bash

      export OMPI_MCA_coll_ibm_skip_barrier=true

.. warning::

   Sep 3rd, 2021 (OLCFHELP-3319):
   If you are an active developer and compile middleware libraries (e.g., ADIOS2) yourself that use MPI and/or infiniband, be aware of ``libfabric``: IBM forks the open source version of this library and ships a patched version.

   Avoid conflicts with mainline versions of this library in MPI that lead to crashes at runtime by loading alongside the system MPI module:

   .. code-block:: bash

      module load libfabric/1.12.1-sysrdma

   For instance, if you compile large software stacks with Spack, make sure to register ``libfabric`` with that exact version as an external module.

   If you load the documented ADIOS2 module above, this problem does not affect you, since the correct ``libfabric`` version is chosen for this one.

.. warning::

   Related to the above issue, the fabric selection in ADIOS2 was designed for libfabric 1.6.
   With newer versions of libfabric, a workaround is needed to guide the selection of a functional fabric for RDMA support.
   Details are discussed in `ADIOS2 issue #2887 <https://github.com/ornladios/ADIOS2/issues/2887>`__.

   The following environment variables can be set as work-arounds, when working with ADIOS2 SST:

   .. code-block:: bash

      export FABRIC_IFACE=mlx5_0   # ADIOS SST: select interface (1 NIC on Summit)
      export FI_OFI_RXM_USE_SRX=1  # libfabric: use shared receive context from MSG provider

.. warning::

   Oct 12th, 2021 (OLCFHELP-4242):
   There is currently a problem with the pre-installed Jupyter extensions, which can lead to connection splits at long running analysis sessions.

   Work-around this issue by running in a single Jupyter cell, before starting analysis:

   .. code-block:: bash

      !jupyter serverextension enable --py --sys-prefix dask_labextension


.. _post-processing-summit:

Post-Processing
---------------

For post-processing, most users use Python via OLCFs's `Jupyter service <https://jupyter.olcf.ornl.gov>`__ (`Docs <https://docs.olcf.ornl.gov/services_and_applications/jupyter/index.html>`__).

We usually just install our software on-the-fly on Summit.
When starting up a post-processing session, run this in your first cells:

.. note::

   The following software packages are installed only into a temporary directory.

.. code-block:: bash

   # work-around for OLCFHELP-4242
   !jupyter serverextension enable --py --sys-prefix dask_labextension

   # next Jupyter cell: the software you want
   !mamba install --quiet -c conda-forge -y openpmd-api openpmd-viewer ipympl ipywidgets fast-histogram yt

   # restart notebook
