.. _building-perlmutter:

Perlmutter (NERSC)
==================

The `Perlmutter cluster <https://docs.nersc.gov/systems/perlmutter/>`_ is located at NERSC.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `NERSC user guide <https://docs.nersc.gov/>`__
* Batch system: `Slurm <https://docs.nersc.gov/systems/perlmutter/#running-jobs>`__
* `Jupyter service <https://jupyter.nersc.gov>`__ (`documentation <https://docs.nersc.gov/services/jupyter/>`__)
* `Filesystems <https://docs.nersc.gov/filesystems/>`__:

  * ``$HOME``: per-user directory, use only for inputs, source and scripts; backed up (40GB)
  * ``${CFS}/m3239/``: `community file system <https://docs.nersc.gov/filesystems/community/>`__ for users in the project ``m3239`` (or equivalent); moderate performance (20TB default)
  * ``$PSCRATCH``: per-user `production directory <https://docs.nersc.gov/filesystems/perlmutter-scratch/>`__; very fast for parallel jobs; purged every 8 weeks (20TB default)


.. _building-perlmutter-preparation:

Preparation
-----------

Use the following commands to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx

On Perlmutter, you can run either on GPU nodes with fast A100 GPUs (recommended) or CPU nodes.

.. tab-set::

   .. tab-item:: A100 GPUs

      We use system software modules, add environment hints and further dependencies via the file ``$HOME/perlmutter_gpu_warpx.profile``.
      Create it now:

      .. code-block:: bash

         cp $HOME/src/warpx/Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example $HOME/perlmutter_gpu_warpx.profile

      .. dropdown:: Script Details
         :color: light
         :icon: info
         :animate: fade-in-slide-down

         .. literalinclude:: ../../../../Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example
            :language: bash

      Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
      Perlmutter GPU projects must end in ``..._g``.
      For example, if you are member of the project ``m3239``, then run ``nano $HOME/perlmutter_gpu_warpx.profile`` and edit line 2 to read:

      .. code-block:: bash

         export proj="m3239_g"

      Exit the ``nano`` editor with ``Ctrl`` + ``O`` (save) and then ``Ctrl`` + ``X`` (exit).

      .. important::

         Now, and as the first step on future logins to Perlmutter, activate these environment settings:

         .. code-block:: bash

            source $HOME/perlmutter_gpu_warpx.profile

      Finally, since Perlmutter does not yet provide software modules for some of our dependencies, install them once:

      .. code-block:: bash

         bash $HOME/src/warpx/Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh
         source ${CFS}/${proj%_g}/${USER}/sw/perlmutter/gpu/venvs/warpx-gpu/bin/activate

      .. dropdown:: Script Details
         :color: light
         :icon: info
         :animate: fade-in-slide-down

         .. literalinclude:: ../../../../Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh
            :language: bash


   .. tab-item:: CPU Nodes

      We use system software modules, add environment hints and further dependencies via the file ``$HOME/perlmutter_cpu_warpx.profile``.
      Create it now:

      .. code-block:: bash

         cp $HOME/src/warpx/Tools/machines/perlmutter-nersc/perlmutter_cpu_warpx.profile.example $HOME/perlmutter_cpu_warpx.profile

      .. dropdown:: Script Details
         :color: light
         :icon: info
         :animate: fade-in-slide-down

         .. literalinclude:: ../../../../Tools/machines/perlmutter-nersc/perlmutter_cpu_warpx.profile.example
            :language: bash

      Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
      For example, if you are member of the project ``m3239``, then run ``nano $HOME/perlmutter_cpu_warpx.profile`` and edit line 2 to read:

      .. code-block:: bash

         export proj="m3239"

      Exit the ``nano`` editor with ``Ctrl`` + ``O`` (save) and then ``Ctrl`` + ``X`` (exit).

      .. important::

         Now, and as the first step on future logins to Perlmutter, activate these environment settings:

         .. code-block:: bash

            source $HOME/perlmutter_cpu_warpx.profile

      Finally, since Perlmutter does not yet provide software modules for some of our dependencies, install them once:

      .. code-block:: bash

         bash $HOME/src/warpx/Tools/machines/perlmutter-nersc/install_cpu_dependencies.sh
         source ${CFS}/${proj}/${USER}/sw/perlmutter/cpu/venvs/warpx-cpu/bin/activate

      .. dropdown:: Script Details
         :color: light
         :icon: info
         :animate: fade-in-slide-down

         .. literalinclude:: ../../../../Tools/machines/perlmutter-nersc/install_cpu_dependencies.sh
            :language: bash


.. _building-perlmutter-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. tab-set::

   .. tab-item:: A100 GPUs

      .. code-block:: bash

         cd $HOME/src/warpx
         rm -rf build_pm_gpu

         cmake -S . -B build_pm_gpu -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
         cmake --build build_pm_gpu -j 16

      The WarpX application executables are now in ``$HOME/src/warpx/build_pm_gpu/bin/``.
      Additionally, the following commands will install WarpX as a Python module:

      .. code-block:: bash

         cd $HOME/src/warpx
         rm -rf build_pm_gpu_py

         cmake -S . -B build_pm_gpu_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
         cmake --build build_pm_gpu_py -j 16 --target pip_install

   .. tab-item:: CPU Nodes

      .. code-block:: bash

         cd $HOME/src/warpx
         rm -rf build_pm_cpu

         cmake -S . -B build_pm_cpu -DWarpX_COMPUTE=OMP -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
         cmake --build build_pm_cpu -j 16

      The WarpX application executables are now in ``$HOME/src/warpx/build_pm_cpu/bin/``.
      Additionally, the following commands will install WarpX as a Python module:

      .. code-block:: bash

         rm -rf build_pm_cpu_py

         cmake -S . -B build_pm_cpu_py -DWarpX_COMPUTE=OMP -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
         cmake --build build_pm_cpu_py -j 16 --target pip_install

Now, you can :ref:`submit Perlmutter compute jobs <running-cpp-perlmutter>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit Perlmutter jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-cpp-perlmutter>` or copy them to a location in ``$PSCRATCH``.


.. _building-perlmutter-update:

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

- :ref:`update the perlmutter_gpu_warpx.profile or perlmutter_cpu_warpx files <building-perlmutter-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-perlmutter-preparation>`.

As a last step, clean the build directory ``rm -rf $HOME/src/warpx/build_pm_*`` and rebuild WarpX.


.. _running-cpp-perlmutter:

Running
-------

.. tab-set::

   .. tab-item:: A100 (40GB) GPUs

      The batch script below can be used to run a WarpX simulation on multiple nodes (change ``-N`` accordingly) on the supercomputer Perlmutter at NERSC.
      This partition as up to `1536 nodes <https://docs.nersc.gov/systems/perlmutter/architecture/>`__.

      Replace descriptions between chevrons ``<>`` by relevant values, for instance ``<input file>`` could be ``plasma_mirror_inputs``.
      Note that we run one MPI rank per GPU.

      .. literalinclude:: ../../../../Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch
         :language: bash
         :caption: You can copy this file from ``$HOME/src/warpx/Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch``.

      To run a simulation, copy the lines above to a file ``perlmutter_gpu.sbatch`` and run

      .. code-block:: bash

         sbatch perlmutter_gpu.sbatch

      to submit the job.


   .. tab-item:: A100 (80GB) GPUs

      Perlmutter has `256 nodes <https://docs.nersc.gov/systems/perlmutter/architecture/>`__ that provide 80 GB HBM per A100 GPU.
      In the A100 (40GB) batch script, replace ``-C gpu`` with ``-C gpu&hbm80g`` to use these large-memory GPUs.


   .. tab-item:: CPU Nodes

      The Perlmutter CPU partition as up to `3072 nodes <https://docs.nersc.gov/systems/perlmutter/architecture/>`__, each with 2x AMD EPYC 7763 CPUs.

      .. literalinclude:: ../../../../Tools/machines/perlmutter-nersc/perlmutter_cpu.sbatch
         :language: bash
         :caption: You can copy this file from ``$HOME/src/warpx/Tools/machines/perlmutter-nersc/perlmutter_cpu.sbatch``.


.. _post-processing-perlmutter:

Post-Processing
---------------

For post-processing, most users use Python via NERSC's `Jupyter service <https://jupyter.nersc.gov>`__ (`documentation <https://docs.nersc.gov/services/jupyter/>`__).

As a one-time preparatory setup, log into Perlmutter via SSH and do *not* source the WarpX profile script above.
Create your own Conda environment and `Jupyter kernel <https://docs.nersc.gov/services/jupyter/how-to-guides/#how-to-use-a-conda-environment-as-a-python-kernel>`__ for post-processing:

.. code-block:: bash

   module load python

   conda config --set auto_activate_base false

   # create conda environment
   rm -rf $HOME/.conda/envs/warpx-pm-postproc
   conda create --yes -n warpx-pm-postproc -c conda-forge mamba conda-libmamba-solver
   conda activate warpx-pm-postproc
   conda config --set solver libmamba
   mamba install --yes -c conda-forge python ipykernel ipympl matplotlib numpy pandas yt openpmd-viewer openpmd-api h5py fast-histogram dask dask-jobqueue pyarrow

   # create Jupyter kernel
   rm -rf $HOME/.local/share/jupyter/kernels/warpx-pm-postproc/
   python -m ipykernel install --user --name warpx-pm-postproc --display-name WarpX-PM-PostProcessing
   echo -e '#!/bin/bash\nmodule load python\nsource activate warpx-pm-postproc\nexec "$@"' > $HOME/.local/share/jupyter/kernels/warpx-pm-postproc/kernel-helper.sh
   chmod a+rx $HOME/.local/share/jupyter/kernels/warpx-pm-postproc/kernel-helper.sh
   KERNEL_STR=$(jq '.argv |= ["{resource_dir}/kernel-helper.sh"] + .' $HOME/.local/share/jupyter/kernels/warpx-pm-postproc/kernel.json | jq '.argv[1] = "python"')
   echo ${KERNEL_STR} | jq > $HOME/.local/share/jupyter/kernels/warpx-pm-postproc/kernel.json

   exit


When opening a Jupyter notebook on `https://jupyter.nersc.gov <https://jupyter.nersc.gov>`__, just select ``WarpX-PM-PostProcessing`` from the list of available kernels on the top right of the notebook.

Additional software can be installed later on, e.g., in a Jupyter cell using ``!mamba install -y -c conda-forge ...``.
Software that is not available via conda can be installed via ``!python -m pip install ...``.
