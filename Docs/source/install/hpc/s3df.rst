.. _building-perlmutter:

S3DF (SLAC)
==================

The `SLAC Shared Scientific Data Facility (S3DF) cluster <https://s3df.slac.stanford.edu/#/>`_ is located at SLAC.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `S3DF user guide <https://usatlas.readthedocs.io/projects/af-docs/en/latest/sshlogin/ssh2SLAC/#accessing-to-s3df>`__
* Batch system: `Slurm <https://s3df.slac.stanford.edu/#/batch-compute?id=slurm>`__
* `Jupyter service <https://usatlas.readthedocs.io/projects/af-docs/en/latest/jupyter/SLACjupyter/>`__
* `Filesystems <https://s3df.slac.stanford.edu/#/reference?id=backup>`__:

  * ``$HOME``: sdfhome, per-user directory, default `quota <https://s3df.slac.stanford.edu/#/reference?id=storagequota>`__: 30GB
  * per-project directory, 10 TB

.. _building-s3df-preparation:

Preparation
-----------

After connecting to the S3DF log in to one of the pools. You can find a list of available pools `here <https://s3df.slac.stanford.edu/#/interactive-compute?id=using-a-terminal>`__. Here we will install WarpX on the ``iana`` pool:

.. code-block:: bash

   ssh iana

.. _building-s3df-login:

It is recommended that you install WarpX in your project directory instead of your ``$HOME`` directory. Navigate there with:

.. code-block:: bash

   cd /sdf/group/projectname/username

Here replace ``projectname`` with your project e.g. ``facet``, and ``username`` with your s3df user name. It makes sense to save this path for frequent use by putting it in your ``.bashrc`` file:

.. code-block:: bash

    vi ~/.bashrc

Add the following line in this file:

.. code-block:: bash

   export WORK="/sdf/group/projectname/username"

Quit the file and run it:

.. code-block:: bash

    source ~/.bashrc

Like this the ``WORK`` variable will be automatically set at every login.

Use the following command to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/BLAST-WarpX/warpx.git $WORK/src/warpx

.. _building-s3df-init

On S3DF, you can run either on GPU nodes with fast A100 GPUs (recommended) or CPU nodes.

.. tab-set::

   .. tab-item:: A100 GPUs

      We use system software modules, add environment hints and further dependencies via the file ``$WORK/s3df_gpu_warpx.profile``.
      Create it now:

      .. code-block:: bash

         cp $WORK/src/warpx/Tools/machines/s3df-slac/s3df_gpu_warpx.profile.example $WORK/s3df_gpu_warpx.profile

      .. dropdown:: Script Details
         :color: light
         :icon: info
         :animate: fade-in-slide-down

         .. literalinclude:: ../../../../Tools/machines/s3df-slac/s3df_gpu_warpx.profile.example
            :language: bash

      Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
      For example, if you are member of the project ``facet``, then run ``nano $WORK/s3df_gpu_warpx.profile`` and edit line 2 to read:

      .. code-block:: bash

         export proj="facet"

      Exit the ``nano`` editor with ``Ctrl`` + ``O`` (save) and then ``Ctrl`` + ``X`` (exit).

      .. important::

         Now, and as the first step on future logins to Perlmutter, activate these environment settings:

         .. code-block:: bash
            cd $WORK
            source $WORK/s3df_gpu_warpx.profile

      Finally, since Perlmutter does not yet provide software modules for some of our dependencies, install them once:

      .. code-block:: bash

         bash $WORK/src/warpx/Tools/machines/s3df-slac/install_gpu_dependencies.sh
         source $WORK/sw/warpx/s3df/gpu/venvs/warpx-gpu/bin/activate

      .. dropdown:: Script Details
         :color: light
         :icon: info
         :animate: fade-in-slide-down

         .. literalinclude:: ../../../../Tools/machines/s3df-slac/install_gpu_dependencies.sh
            :language: bash


   .. tab-item:: CPU Nodes

      We use system software modules, add environment hints and further dependencies via the file ``$WORK/s3df_cpu_warpx.profile``.
      Create it now:

      .. code-block:: bash

         cp $WORK/src/warpx/Tools/machines/s3df-slac/s3df_cpu_warpx.profile.example $WORK/s3df_cpu_warpx.profile

      .. dropdown:: Script Details
         :color: light
         :icon: info
         :animate: fade-in-slide-down

         .. literalinclude:: ../../../../Tools/machines/s3df-slac/s3df_cpu_warpx.profile.example
            :language: bash

      Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
      For example, if you are member of the project ``facet``, then run ``nano $WORK/s3df_cpu_warpx.profile`` and edit line 2 to read:

      .. code-block:: bash

         export proj="facet"

      Exit the ``nano`` editor with ``Ctrl`` + ``O`` (save) and then ``Ctrl`` + ``X`` (exit).

      .. important::

         Now, and as the first step on future logins to S3DF, activate these environment settings:

         .. code-block:: bash

            source $WORK/s3df_cpu_warpx.profile

      Finally, since S3DF does not yet provide software modules for some of our dependencies, install them once:

      .. code-block:: bash

         bash $WORK/src/warpx/Tools/machines/s3df-slac/install_cpu_dependencies.sh
         source $WORK/sw/warpx/s3df/cpu/venvs/warpx-cpu/bin/activate

      .. dropdown:: Script Details
         :color: light
         :icon: info
         :animate: fade-in-slide-down

         .. literalinclude:: ../../../../Tools/machines/s3df-slac/install_cpu_dependencies.sh
            :language: bash


.. _building-perlmutter-compilation:


Compilation
-----------

S3DF pools may miss some of the packages needed for installing WarpX. The best practice is to install them manually into your user file system under a new folder called ``opt``.
The version of ``cmake`` currently provided by S3DF on the ``iana`` pool is 3.20.2 but WarpX requires 3.24.0 or higher. We need to install a newer ``cmake`` manually:

In ``CMakeLists.txt`` you can either toggle the relevant options ON/OFF:
    
.. code-block:: python
    
      # Options and Variants ########################################################
      #
      include(CMakeDependentOption)
      option(WarpX_APP           "Build the WarpX executable application"     ON)
      option(WarpX_ASCENT        "Ascent in situ diagnostics"                 OFF)
      option(WarpX_CATALYST      "Catalyst in situ diagnostics"               OFF)
      option(WarpX_EB            "Embedded boundary support"                  ON)
      option(WarpX_LIB           "Build WarpX as a library"                   OFF)
      option(WarpX_MPI           "Multi-node support (message-passing)"       ON)
      option(WarpX_SIMD          "CPU SIMD Acceleration"                      OFF)
      option(WarpX_OPENPMD       "openPMD I/O (HDF5, ADIOS)"                  ON)
      option(WarpX_FASTMATH      "Enable fast-math optimizations"             OFF)
      option(WarpX_FFT           "FFT-based solvers"                          ON)
      option(WarpX_PYTHON        "Python bindings"                            OFF)
      option(WarpX_SENSEI        "SENSEI in situ diagnostics"                 OFF)
      option(WarpX_QED           "QED support (requires PICSAR)"              ON)
      option(WarpX_QED_TABLE_GEN "QED table generation (requires PICSAR and Boost)"
                                                                              OFF)
      option(WarpX_QED_TOOLS     "Build external tool to generate QED lookup tables (requires PICSAR and Boost)"
                                                                              OFF)
      
      # Advanced option to run tests
      option(WarpX_TEST_CLEANUP "Clean up automated test directories" OFF)
      option(WarpX_TEST_DEBUGGER "Run automated tests without AMReX signal handling (to attach debuggers)" OFF)
      option(WarpX_TEST_FPETRAP "Run automated tests with FPE-trapping runtime parameters" OFF)
    
    or append the desired options in the format ``cmake -S . -B build -DWarpX_FFT=ON -WarpX_QED=ON`` when calling the build command.

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. tab-set::

   .. tab-item:: A100 GPUs

      .. code-block:: bash

         cd $WORK/src/warpx
         rm -rf build_pm_gpu

         cmake -S . -B build_pm_gpu -DWarpX_COMPUTE=CUDA -DWarpX_PSATD=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_LIB=ON -DWarpX_DIMS="1;2;RZ;3"
         cmake --build build_pm_gpu -j 16

    The WarpX application executables are now in $WORK/src/warpx/build_pm_gpu/bin/. Additionally, the following commands will install WarpX as a Python module:

    .. code-block:: bash
        rm -rf build_pm_gpu_py

        cmake -S . -B build_pm_gpu_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
        cmake --build build_pm_gpu_py -j 16 --target pip_install

   .. tab-item:: CPU Nodes

      .. code-block:: bash

         cd $WORK/src/warpx
         rm -rf build_pm_cpu

         cmake -S . -B build_pm_cpu -DWarpX_COMPUTE=OMP -DWarpX_PSATD=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_LIB=ON -DWarpX_DIMS="1;2;RZ;3"
         cmake --build build_pm_cpu -j 16

    The WarpX application executables are now in $WORK/src/warpx/build_pm_cpu/bin/. Additionally, the following commands will install WarpX as a Python module:
      
    .. code-block:: bash
        rm -rf build_pm_cpu_py

        cmake -S . -B build_pm_cpu_py -DWarpX_COMPUTE=OMP -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
        cmake --build build_pm_cpu_py -j 16 --target pip_install

.. _building-s3df-update:

Update WarpX & Dependencies
---------------------------

If you already installed WarpX in the past and want to update it, start by getting the latest source code:

.. code-block:: bash

   cd $WORK/src/warpx

   # read the output of this command - does it look ok?
   git status

   # get the latest WarpX source code
   git fetch
   git pull

   # read the output of these commands - do they look ok?
   git status
   git log # press q to exit

.. _update-s3df-update:

And, if needed,

- :ref:`update the s3df_warpx files <building-s3df-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-s3df-preparation>`.

As a last step, clean the build directory ``rm -rf $WORK/src/warpx/build_pm_*`` and rebuild WarpX.

Running
-------

For general instructions refer to `here <https://warpx.readthedocs.io/en/latest/usage/how_to_run.html#run>`__.
The ``iana`` pool gives access to 4 servers, 40 HT cores and 384 GB per server (see `here <https://s3df.slac.stanford.edu/#/interactive-compute?id=using-a-terminal>`__).

.. tab-set::

   .. tab-item:: A100 GPUs

        You can run an interactive job directly from the terminal by typing:

        .. code-block:: bash
        
            srun --partition <partitionname> --account <accountname> -N 2 -n 2 -c 4 --gres=gpu:1 --time=01:00:00 --pty /bin/bash
        
        .. _running-s3df-terminal:

        where ``<partitionname>`` is the cluster partition that you want to use (see `here <https://s3df.slac.stanford.edu/#/batch-compute?id=partitions-amp-accounts>`__),  and ``<accountname>`` is the group you are associated with e.g. ``facet``. Note that on S3DF there are no individual accounts, only group accounts. Here we request ``-N 1`` nodes, ``-n 1`` tasks (MPI processes) and ``-c 1`` CPU cores (physical threads) per task and and one GPU (``--gres=gpu:1``). Once the scheduler allocated the resources you can run WarpX:

        .. code-block:: bash
        
            module load openmpi/v4.1.6
            srun ./warpx.3d.MPI.CUDA.DP.PDP.OPMD.FFT.EB.QED.GENQEDTABLES input_simple.txt
        
        .. _running-s3df-run:

        Alternatively you can create a bash submission script ``submit.sh`` with the content shown below and launch it with ``sbatch submit_gpu.sh``.
        
        .. code-block:: bash
        
            #!/bin/bash -l
            
            # Copyright 2025 Peter Kicsiny
            #
            # This file is part of WarpX.
            #
            # License: BSD-3-Clause-LBNL
            
            #SBATCH --partition=milano
            #SBATCH --account=facet
            #SBATCH -N 1 # --nodes= number of nodes
            #SBATCH --gres=gpu:1 # request a GPU
            #SBATCH -t 00:10:00 # --time= walltime
            #SBATCH -J WarpX # --job-name= name of job
            #SBATCH --qos=preemptable # queue
            #SBATCH -o WarpX.o%j # --output= stdout
            #SBATCH -e WarpX.e%j # --error= stderr
            
            # executable & inputs file or python interpreter & PICMI script here
            EXE=./warpx.3d.MPI.CUDA.DP.PDP.OPMD.FFT.EB.QED.GENQEDTABLES
            INPUTS=input_simple.txt

            command="module load openmpi/v4.1.6"
            echo ${command}
            ${command}
            
            command="srun ${EXE} ${INPUTS}"
            echo ${command}
            ${command}

        .. _running-s3df-script:

        Here an example is shown using the group account ``facet`` and the ``preemptable`` queue. 

   .. tab-item:: CPU Nodes
        
        You can run an interactive job directly from the terminal by typing:

        .. code-block:: bash
        
            srun --partition <partitionname> --account <accountname> -N 2 -n 2 -c 4 --time=01:00:00 --pty /bin/bash
        
        .. _running-s3df-terminal:
        
        where ``<partitionname>`` is the cluster partition that you want to use (see `here <https://s3df.slac.stanford.edu/#/batch-compute?id=partitions-amp-accounts>`__),  and ``<accountname>`` is the group you are associated with e.g. ``facet``. Note that on S3DF there are no individual accounts, only group accounts. Here we request ``-N 2`` nodes, ``-n 2`` tasks (MPI processes) and ``-c 4`` CPU cores (physical threads) per task. This means 8 CPU cores will be requested by this job, 1 job will run per node, each job with 4 CPU cores. Once the scheduler allocated the resources you can run WarpX:
        
        .. code-block:: bash
        
            module load openmpi/v4.1.6
            mpirun -np 2 ./warpx.3d.MPI.OMP.DP.PDP.OPMD.FFT.EB.QED input_simple.txt
        
        .. _running-s3df-run:

        Alternatively you can create a bash submission script ``submit.sh`` with the content shown below and launch it like ``sbatch submit_cpu.sh``.
        
        .. code-block:: bash
        
            #!/bin/bash -l
            
            # Copyright 2025 Peter Kicsiny
            #
            # This file is part of WarpX.
            #
            # License: BSD-3-Clause-LBNL
            
            #SBATCH --partition=milano
            #SBATCH --account=facet
            #SBATCH -N 2 # --nodes= number of nodes
            #SBATCH -n 2 # --ntasks= number of MPI processes or tasks in total
            #SBATCH -c 4 # --cpus-per-task= number of CPU cores (threads) per task
            #SBATCH -t 00:10:00 # --time= walltime
            #SBATCH -J WarpX # --job-name= name of job
            #SBATCH --qos=preemptable # queue
            #SBATCH -o WarpX.o%j # --output= stdout
            #SBATCH -e WarpX.e%j # --error= stderr
            
            # executable & inputs file or python interpreter & PICMI script here
            EXE=./warpx.3d.MPI.OMP.DP.PDP.OPMD.FFT.EB.QED
            INPUTS=input_simple.txt
            
            # threads for OpenMP and threaded compressors per MPI rank
            export OMP_PLACES=threads
            export OMP_PROC_BIND=spread
            export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} # -c 4 so 4 threads per MPI rank
            
            command="module load openmpi/v4.1.6"
            echo ${command}
            ${command}
            
            command="mpirun -np ${SLURM_NTASKS} ${EXE} ${INPUTS}"
            echo ${command}
            ${command}

        .. _running-s3df-script:

        Here an example is shown using the group account ``facet`` and the ``preemptable`` queue. We request 2 nodes and run 2 MPI ranks each using 4 threads. 
