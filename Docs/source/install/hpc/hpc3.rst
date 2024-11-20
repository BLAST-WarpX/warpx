.. _building-hpc3:

HPC3 (UCI)
==========

The `HPC3 supercomputer <https://rcic.uci.edu/hpc3/index.html>`_ is located at University of California, Irvine.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `HPC3 user guide <https://rcic.uci.edu/hpc3/index.html>`__
* Batch system: `Slurm <https://rcic.uci.edu/hpc3/slurm.html>`__ (`notes <https://rcic.uci.edu/hpc3/examples.html#_submit_different_job_types>`__)
* `Jupyter service <https://rcic.uci.edu/hpc3/examples.html#jupyterhub-portal>`__
* `Filesystems <https://rcic.uci.edu/storage/beegfs-howtos.html>`__:

  * ``$HOME``: per-user directory, use only for inputs, source and scripts; backed up (40GB)
  * ``/pub/$USER``: per-user production directory; fast and larger storage for parallel jobs (1TB default quota)
  * ``/dfsX/<lab-path>`` lab group quota (based on PI's purchase allocation). The storage owner (PI) can specify what users have read/write capability on the specific filesystem.


.. _building-hpc3-preparation:

Preparation
-----------

Use the following commands to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx

On HPC3, you recommend to run on the `fast GPU nodes with V100 GPUs <https://rcic.uci.edu/hpc3/slurm.html#memmap>`__.

We use system software modules, add environment hints and further dependencies via the file ``$HOME/hpc3_gpu_warpx.profile``.
Create it now:

.. code-block:: bash

  cp $HOME/src/warpx/Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example $HOME/hpc3_gpu_warpx.profile

.. dropdown:: Script Details
  :color: light
  :icon: info
  :animate: fade-in-slide-down

  .. literalinclude:: ../../../../Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example
     :language: bash

Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
For example, if you are member of the project ``plasma``, then run ``vi $HOME/hpc3_gpu_warpx.profile``.
Enter the edit mode by typing ``i`` and edit line 2 to read:

.. code-block:: bash

   export proj="plasma"

Exit the ``vi`` editor with ``Esc`` and then type ``:wq`` (write & quit).

.. important::

   Now, and as the first step on future logins to HPC3, activate these environment settings:

   .. code-block:: bash

      source $HOME/hpc3_gpu_warpx.profile

Finally, since HPC3 does not yet provide software modules for some of our dependencies, install them once:

.. code-block:: bash

   bash $HOME/src/warpx/Tools/machines/hpc3-uci/install_gpu_dependencies.sh
   source $HOME/sw/hpc3/gpu/venvs/warpx-gpu/bin/activate

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/hpc3-uci/install_gpu_dependencies.sh
      :language: bash


.. _building-hpc3-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build

   cmake -S . -B build -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build -j 8

The WarpX application executables are now in ``$HOME/src/warpx/build/bin/``.
Additionally, the following commands will install WarpX as a Python module:

.. code-block:: bash

   rm -rf build_py

   cmake -S . -B build_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_py -j 8 --target pip_install

Now, you can :ref:`submit HPC3 compute jobs <running-cpp-hpc3>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit HPC3 jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-cpp-hpc3>` or copy them to a location in ``$PSCRATCH``.


.. _building-hpc3-update:

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

- :ref:`update the hpc3_gpu_warpx.profile file <building-hpc3-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-hpc3-preparation>`.

As a last step, clean the build directory ``rm -rf $HOME/src/warpx/build`` and rebuild WarpX.


.. _running-cpp-hpc3:

Running
-------

The batch script below can be used to run a WarpX simulation on multiple nodes (change ``-N`` accordingly) on the supercomputer HPC3 at UCI.
This partition as up to `32 nodes <https://rcic.uci.edu/hpc3/slurm.html#memmap>`__ with four V100 GPUs (16 GB each) per node.

Replace descriptions between chevrons ``<>`` by relevant values, for instance ``<proj>`` could be ``plasma``.
Note that we run one MPI rank per GPU.

.. literalinclude:: ../../../../Tools/machines/hpc3-uci/hpc3_gpu.sbatch
   :language: bash
   :caption: You can copy this file from ``$HOME/src/warpx/Tools/machines/hpc3-uci/hpc3_gpu.sbatch``.

To run a simulation, copy the lines above to a file ``hpc3_gpu.sbatch`` and run

.. code-block:: bash

   sbatch hpc3_gpu.sbatch

to submit the job.


.. _post-processing-hpc3:

Post-Processing
---------------

UCI provides a pre-configured `Jupyter service <https://rcic.uci.edu/hpc3/examples.html#jupyterhub-portal>`__ that can be used for data-analysis.

We recommend to install at least the following ``pip`` packages for running Python3 Jupyter notebooks on WarpX data analysis:
``h5py ipympl ipywidgets matplotlib numpy openpmd-viewer openpmd-api pandas scipy yt``
