.. _building-tuolumne:

Tuolumne (LLNL)
===============

The `Tuolumne AMD GPU cluster <https://hpc.llnl.gov/hardware/compute-platforms/tuolumne>`__ (short name *tuo*) is located at LLNL.
Tuolumne is an unclassified sibling system of `El Capitan <https://hpc.llnl.gov/hardware/compute-platforms/el-capitan>`__, sharing the same architecture.

El Capitan & Tuolumne provide four AMD MI300A APUs per compute node.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `Tuolumne overview <https://hpc.llnl.gov/hardware/compute-platforms/tuolumne>`__
* `LLNL user account <https://lc.llnl.gov/lorenz/mylc/mylc.cgi>`__ (login required)
* `Tuolumne user guide <https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems>`__
* Batch system: `Flux with Slurm Wrappers <https://lc.llnl.gov/confluence/display/ELCAPEA/Running+Jobs>`__
* `Jupyter service <https://lc.llnl.gov/jupyter>`__ (`documentation <https://lc.llnl.gov/confluence/display/LC/JupyterHub+and+Jupyter+Notebook>`__, login required)
* `Production directories <https://lc.llnl.gov/confluence/display/ELCAPEA/File+Systems>`__ (login required):

  * ``/p/lustre5/${USER}``: personal directory on the parallel filesystem (also: ``lustre2``)
  * Note that the ``$HOME`` directory and the ``/usr/workspace/${USER}`` space are NFS mounted and *not* suitable for production quality data generation.


Login
-----

.. code-block:: bash

   ssh tuolumne.llnl.gov


.. _building-tuolumne-preparation:

Preparation
-----------

Use the following commands to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/BLAST-WarpX/warpx.git /p/lustre5/${USER}/tuolumne/src/warpx

We use system software modules, add environment hints and further dependencies via the file ``$HOME/tuolumne_mi300a_warpx.profile``.
Create it now:

.. code-block:: bash

   cp /p/lustre5/${USER}/tuolumne/src/warpx/Tools/machines/tuolumne-llnl/tuolumne_mi300a_warpx.profile.example $HOME/tuolumne_mi300a_warpx.profile

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/tuolumne-llnl/tuolumne_mi300a_warpx.profile.example
      :language: bash

Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
**Currently, this is unused and can be kept empty.**
Once project allocation becomes required, e.g., if you are member of the project ``abcde``, then run ``vi $HOME/tuolumne_mi300a_warpx.profile``.
Enter the edit mode by typing ``i`` and edit line 2 to read:

.. code-block:: bash

   export proj="abcde"

Exit the ``vi`` editor with ``Esc`` and then type ``:wq`` (write & quit).

.. important::

   Now, and as the first step on future logins to Tuolumne, activate these environment settings:

   .. code-block:: bash

      source $HOME/tuolumne_mi300a_warpx.profile

Finally, since Tuolumne does not yet provide software modules for some of our dependencies, install them once:


  .. code-block:: bash

     bash /p/lustre5/${USER}/tuolumne/src/warpx/Tools/machines/tuolumne-llnl/install_mi300a_dependencies.sh
     source /p/lustre5/${USER}/tuolumne/warpx/mi300a/venvs/warpx-tuolumne-mi300a/bin/activate

  .. dropdown:: Script Details
     :color: light
     :icon: info
     :animate: fade-in-slide-down

     .. literalinclude:: ../../../../Tools/machines/tuolumne-llnl/install_mi300a_dependencies.sh
        :language: bash

  .. dropdown:: AI/ML Dependencies (Optional)
     :animate: fade-in-slide-down

     If you plan to run AI/ML workflows depending on PyTorch et al., run the next step as well.
     This will take a while and should be skipped if not needed.

     .. code-block:: bash

        bash /p/lustre5/${USER}/tuolumne/src/warpx/Tools/machines/tuolumne-llnl/install_mi300a_ml.sh

     .. dropdown:: Script Details
        :color: light
        :icon: info
        :animate: fade-in-slide-down

        .. literalinclude:: ../../../../Tools/machines/tuolumne-llnl/install_mi300a_ml.sh
           :language: bash


.. _building-tuolumne-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. code-block:: bash

   cd /p/lustre5/${USER}/tuolumne/src/warpx

   cmake --fresh -S . -B build_tuolumne -DWarpX_COMPUTE=HIP -DWarpX_FFT=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_tuolumne -j 24

The WarpX application executables are now in ``/p/lustre5/${USER}/tuolumne/src/warpx/build_tuolumne/bin/``.
Additionally, the following commands will install WarpX as a Python module:

.. code-block:: bash

   cmake --fresh -S . -B build_tuolumne_py -DWarpX_COMPUTE=HIP -DWarpX_FFT=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_tuolumne_py -j 24 --target pip_install

Now, you can :ref:`submit tuolumne compute jobs <running-cpp-tuolumne>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit tuolumne jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-cpp-tuolumne>` or copy them to a location in ``$PROJWORK/$proj/``.


.. _building-tuolumne-update:

Update WarpX & Dependencies
---------------------------

If you already installed WarpX in the past and want to update it, start by getting the latest source code:

.. code-block:: bash

   cd /p/lustre5/${USER}/tuolumne/src/warpx

   # read the output of this command - does it look ok?
   git status

   # get the latest WarpX source code
   git fetch
   git pull

   # read the output of these commands - do they look ok?
   git status
   git log     # press q to exit

And, if needed,

- :ref:`update the tuolumne_mi300a_warpx.profile file <building-tuolumne-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-tuolumne-preparation>`.

As a last step :ref:`rebuild WarpX <building-tuolumne-compilation>`.


.. _running-cpp-tuolumne:

Running
-------

.. _running-cpp-tuolumne-MI300A-APUs:

MI300A APUs (128GB)
^^^^^^^^^^^^^^^^^^^

`Each compute node <https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/introduction-and-quickstart/pro-tips>`__ is divided into 4 sockets, each with:

* 1 MI300A GPU,
* 21 available user CPU cores, with 3 cores reserved for the OS (2 hardware threads per core)
* 128GB HBM3 memory (a single NUMA domain)

The batch script below can be used to run a WarpX simulation on 1 node with 4 APUs on the supercomputer Tuolumne at LLNL.
Replace descriptions between chevrons ``<>`` by relevant values, for instance ``<input file>`` could be ``plasma_mirror_inputs``.
WarpX runs with one MPI rank per GPU.

Note that we append these non-default runtime options:

* ``amrex.use_gpu_aware_mpi=1``: make use of fast APU to APU MPI communications

.. literalinclude:: ../../../../Tools/machines/tuolumne-llnl/tuolumne_mi300a.sbatch
   :language: bash
   :caption: You can copy this file from ``Tools/machines/tuolumne-llnl/tuolumne_mi300a.sbatch``.

To run a simulation, copy the lines above to a file ``tuolumne_mi300a.sbatch`` and run

.. code-block:: bash

   sbatch tuolumne_mi300a.sbatch

to submit the job.
