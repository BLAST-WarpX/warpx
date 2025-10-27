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

.. _building-s3df-compilation:

Use the following command to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/BLAST-WarpX/warpx.git $HOME/src/warpx

.. _building-s3df-compilation:

Compilation
-----------

S3DF is missing some of the packages needed for installing WarpX. The best practice is to install them manually into your user file system under ``~/.local``.
The version of ``cmake`` currently provided by S3DF is 3.20.2 but WarpX requires 3.24.0 or higher. We need to install a newer ``cmake`` manually:

.. code-block:: bash

   cd 
   wget https://github.com/Kitware/CMake/releases/download/v4.2.0-rc1/cmake-4.2.0-rc1-linux-x86_64.sh
   bash cmake-4.2.0-rc1-linux-x86_64.sh --skip-license --prefix=$HOME/.local
   export PATH=$HOME/.local/bin:$PATH
   cmake --version

.. _building-s3df-cmake:

the last command should print ``cmake version 4.2.0-rc1``.

There are additional packages required to compile WarpX such as openmpi to enable MPI support. The general recipe to locally install such backages under `~/.local` is:

.. code-block:: bash

   cd 
   wget <url-to-package>
   cd <package>
   ./configure --prefix=$HOME/.local
   make -j$(nproc)
   make install

.. _building-s3df-packages:

.. code-block:: bash

   export PATH=$HOME/.local/bin:$PATH
   export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
   export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH

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

or append the options in the format ``cmake -S . -B build -DWarpX_FFT=ON -WarpX_QED=ON`` when calling the build command.

Build and compile WarpX:

.. code-block:: bash

   cmake -S . -B build
   cmake --build build -j$(nproc)

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

And, if needed,

- :ref:`update the s3df_warpx files <building-s3df-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-s3df-preparation>`.

As a last step, clean the build directory ``rm -rf $HOME/src/warpx/build_pm_*`` and rebuild WarpX.


.. _running-cpp-s3df:

Running
-------

For gemeral instructions refer to `here <https://warpx.readthedocs.io/en/latest/usage/how_to_run.html#run>`__.

On CPU Nodes

The Perlmutter CPU partition as up to `3072 nodes <https://docs.nersc.gov/systems/perlmutter/architecture/>`__, each with 2x AMD EPYC 7763 CPUs.

.. literalinclude:: ../../../../Tools/machines/perlmutter-nersc/perlmutter_cpu.sbatch
  :language: bash
  :caption: You can copy this file from ``$HOME/src/warpx/Tools/machines/perlmutter-nersc/perlmutter_cpu.sbatch``.

.. _post-processing-s3df:

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
   echo -e '#!/bin/bash\nmodule load python\nconda activate warpx-pm-postproc\nexec "$@"' > $HOME/.local/share/jupyter/kernels/warpx-pm-postproc/kernel-helper.sh
   chmod a+rx $HOME/.local/share/jupyter/kernels/warpx-pm-postproc/kernel-helper.sh
   KERNEL_STR=$(jq '.argv |= ["{resource_dir}/kernel-helper.sh"] + .' $HOME/.local/share/jupyter/kernels/warpx-pm-postproc/kernel.json | jq '.argv[1] = "python"')
   echo ${KERNEL_STR} | jq > $HOME/.local/share/jupyter/kernels/warpx-pm-postproc/kernel.json

   exit


When opening a Jupyter notebook on `https://jupyter.nersc.gov <https://jupyter.nersc.gov>`__, just select ``WarpX-PM-PostProcessing`` from the list of available kernels on the top right of the notebook.

Additional software can be installed later on, e.g., in a Jupyter cell using ``!mamba install -y -c conda-forge ...``.
Software that is not available via conda can be installed via ``!python -m pip install ...``.
