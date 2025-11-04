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

.. _building-s3df-preparation:

Preparation
-----------

After connecting to the S3DF log in to one of the pools. You can find a list of available pools `here <https://s3df.slac.stanford.edu/#/interactive-compute?id=using-a-terminal>`__. Here we will install WarpX on the ``iana`` pool:

.. code-block:: bash

   ssh iana

.. _building-s3df-login:

Use the following command to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/BLAST-WarpX/warpx.git $HOME/src/warpx

.. _building-s3df-init:

Compilation
-----------

S3DF pools may miss some of the packages needed for installing WarpX. The best practice is to install them manually into your user file system under ``~/.local``.
The version of ``cmake`` currently provided by S3DF on the ``iana`` pool is 3.20.2 but WarpX requires 3.24.0 or higher. We need to install a newer ``cmake`` manually:

.. code-block:: bash

   cd 
   wget https://github.com/Kitware/CMake/releases/download/v4.1.2/cmake-4.1.2.tar.gz
   tar -xzf cmake-4.1.2.tar.gz
   cd cmake-4.1.2.tar.gz
   ./bootstrap --prefix=$HOME/opt/cmake-4.1.2
   make -j$(nproc)
   make install
   export PATH=$HOME/opt/cmake-4.2.0/bin:$PATH
   cmake --version

.. _building-s3df-cmake:

the last command should print ``cmake version 4.1.2``.

S3DF uses the `Lmod Module <https://lmod.readthedocs.io/en/latest/010_user.html>`__ system to administrate common software packages not found in the usual software repositories. You can find more information `here <https://s3df.slac.stanford.edu/#/software?id=s3df-centrally-installed-software>`__. WarpX relies on MPI for communicating between compute nodes. We can use the distribution preinstalled on S3DF:

.. code-block:: bash

    module avail openmpi
    module load openmpi/v4.1.6

.. _building-s3df-mpi:

**Make sure you have this MPI loaded when you want to install/run WarpX.** There are additional packages required to successfully compile WarpX such as ``fftw`` ``ADIOS2`` and ``openPMD``. They can be installed in a similar way to ``cmake``. The general recipe to locally install such packages manually, under `~/opt` is:

.. code-block:: bash

    cd 
    wget <url-to-package>
    cd <package>
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/opt/<package-name> -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DENABLE_THREADS=ON
    make -j$(nproc)
    make install

.. _building-s3df-packages:

It is necessary to provide the option ``-DCMAKE_INSTALL_PREFIX=$HOME/opt/<package-name>`` to cmake, otherwise ``make install`` will try to install the package into ``/usr/local/lib64/cmake/`` which will fail if you don't have sudo rights. When the option is provided ``make install`` will create the directory ``opt`` in your home directory and install the packages there. It is conventional to use the package name in lowercase letters and the version number: ``ADIOS2-2.10.1``-> ``<package-name>=adios2-2.10.1`` or ``openPMD-api-0.15.1``-> ``<package-name>=openpmd-0.15.1``. After this we have to add the path of these installed packages to some environment variables so that WarpX finds them during installation. The commands below append the necessary lines into the `~/.bashrc` file so that we don't have to manually add them every time we login to the ``iana`` cluster.

.. code-block:: bash

    echo 'export ADIOS2_DIR=$HOME/opt/adios2' >> ~/.bashrc
    echo 'export FFTW_ROOT=$HOME/opt/fftw' >> ~/.bashrc
    echo 'export PATH=$HOME/opt/adios2/bin:$HOME/opt/fftw/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$HOME/opt/adios2/lib64:$HOME/opt/adios2/lib:$HOME/opt/fftw/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc

.. _building-s3df-paths:

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

CPU Nodes

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build

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

Build and compile WarpX:

.. code-block:: bash

   cmake -S . -B build
   cmake --build build -j$(nproc)

.. _building-s3df-compile:

The WarpX application executables are now in ``$HOME/src/warpx/build/bin/``.
Additionally, the following commands will install WarpX as a Python module:

.. code-block:: bash

    rm -rf build_py

    cmake -S . -B build_pm_py -DWarpX_APP=OFF -DWarpX_PYTHON=ON
    cmake --build build_pm_py -j$(nproc) --target pip_install

.. _building-s3df-update:

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

.. _update-s3df-update:

And, if needed,

- :ref:`update the s3df_warpx files <building-s3df-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-s3df-preparation>`.

As a last step, clean the build directory ``rm -rf $HOME/src/warpx/build_pm_*`` and rebuild WarpX.

Running
-------

For general instructions refer to `here <https://warpx.readthedocs.io/en/latest/usage/how_to_run.html#run>`__.

On CPU Nodes

The ``iana`` pool gives access to 4 servers, 40 HT cores and 384 GB per server (see `here <https://s3df.slac.stanford.edu/#/interactive-compute?id=using-a-terminal>`__).

You can run an interactive job directly from the terminal by typing:

.. code-block:: bash

    srun --partition <partitionname> --account <accountname> -N 2 -n 2 -c 4 --time=01:00:00 --pty /bin/bash

.. _running-s3df-terminal:

where ``<partitionname>`` is the cluster partiction that you want to use (see `here <https://s3df.slac.stanford.edu/#/batch-compute?id=partitions-amp-accounts>`__),  and ``<accountname>`` is the group you are associated with e.g. ``facet``. Note that on S3DF there are no individual accounts, only group accounts. Here we request ``-N 2`` nodes, ``-n 2`` tasks (MPI processes) and ``-c 4`` CPU cores (physical threads) per task. This means 8 CPU cores will be requested by this job, 1 job will run per node, each job with 4 CPU cores. Once the scheduler allocated the resources you can run WarpX:

.. code-block:: bash

    mpirun -np 2 ./warpx.3d.MPI.OMP.DP.PDP.OPMD.FFT.EB.QED input_simple.txt

.. _running-s3df-run:

Alternatively you can create a bash submission script ``submit.sh`` with the content shown below and launch it like ``sbatch submit.sh``.

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
