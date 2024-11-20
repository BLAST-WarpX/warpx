.. _building-adastra:

Adastra (CINES)
===============

The `Adastra cluster <https://www.cines.fr/calcul/adastra/>`_ is located at CINES (France).
Each node contains 4 AMD MI250X GPUs, each with 2 Graphics Compute Dies (GCDs) for a total of 8 GCDs per node.
You can think of the 8 GCDs as 8 separate GPUs, each having 64 GB of high-bandwidth memory (HBM2E).


Introduction
------------

If you are new to this system, **please see the following resources**:

* `Adastra user guide <https://dci.dci-gitlab.cines.fr/webextranet/user_support/>`_
* Batch system: `Slurm <https://dci.dci-gitlab.cines.fr/webextranet/user_support/index.html?highlight=sbatch#running-jobs>`_
* `Production directories <https://dci.dci-gitlab.cines.fr/webextranet/data_and_storage/index.html#data-and-storage>`_:

  * ``$SHAREDSCRATCHDIR``: meant for short-term data storage, shared with all members of a project, purged every 30 days (17.6 TB default quota)
  * ``$SCRATCHDIR``: meant for short-term data storage, single user, purged every 30 days
  * ``$SHAREDWORKDIR``: meant for mid-term data storage, shared with all members of a project, never purged (4.76 TB default quota)
  * ``$WORKDIR``: meant for mid-term data storage, single user, never purged
  * ``$STORE`` : meant for long term storage, single user, never purged, backed up
  * ``$SHAREDHOMEDIR`` : meant for scripts and tools, shared with all members of a project, never purged, backed up
  * ``$HOME`` : meant for scripts and tools, single user, never purged, backed up


.. _building-adastra-preparation:

Preparation
-----------

The following instructions will install WarpX in the ``$SHAREDHOMEDIR`` directory,
which is shared among all the members of a given project. Due to the inode
quota enforced for this machine, a shared installation of WarpX is advised.

Use the following commands to download the WarpX source code:

.. code-block:: bash

   # If you have multiple projects, activate the project that you want to use with:
   #
   # myproject -a YOUR_PROJECT_NAME
   #
   git clone https://github.com/ECP-WarpX/WarpX.git $SHAREDHOMEDIR/src/warpx

We use system software modules, add environment hints and further dependencies via the file ``$SHAREDHOMEDIR/adastra_warpx.profile``.
Create it now:

.. code-block:: bash

   cp $SHAREDHOMEDIR/src/warpx/Tools/machines/adastra-cines/adastra_warpx.profile.example $SHAREDHOMEDIR/adastra_warpx.profile

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/adastra-cines/adastra_warpx.profile.example
      :language: bash

Edit the 2nd line of this script, which sets the ``export proj=""`` variable using a text editor
such as ``nano``, ``emacs``, or ``vim`` (all available by default on Adastra login nodes) and
uncomment the 3rd line (which sets ``$proj`` as the active project).

.. important::

   Now, and as the first step on future logins to Adastra, activate these environment settings:

   .. code-block:: bash

      source $SHAREDHOMEDIR/adastra_warpx.profile

Finally, since Adastra does not yet provide software modules for some of our dependencies, install them once:

.. code-block:: bash

   bash $SHAREDHOMEDIR/src/warpx/Tools/machines/adastra-cines/install_dependencies.sh
   source $SHAREDHOMEDIR/sw/adastra/gpu/venvs/warpx-adastra/bin/activate

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/adastra-cines/install_dependencies.sh
      :language: bash


.. _building-adastra-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. code-block:: bash

   cd $SHAREDHOMEDIR/src/warpx
   rm -rf build_adastra

   cmake -S . -B build_adastra -DWarpX_COMPUTE=HIP -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_adastra -j 16

The WarpX application executables are now in ``$SHAREDHOMEDIR/src/warpx/build_adastra/bin/``.
Additionally, the following commands will install WarpX as a Python module:

.. code-block:: bash

   rm -rf build_adastra_py

   cmake -S . -B build_adastra_py -DWarpX_COMPUTE=HIP -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_adastra_py -j 16 --target pip_install

Now, you can :ref:`submit Adstra compute jobs <running-cpp-adastra>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit Adastra jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-cpp-adastra>` .


.. _building-adastra-update:

Update WarpX & Dependencies
---------------------------

If you already installed WarpX in the past and want to update it, start by getting the latest source code:

.. code-block:: bash

   cd $SHAREDHOMEDIR/src/warpx

   # read the output of this command - does it look ok?
   git status

   # get the latest WarpX source code
   git fetch
   git pull

   # read the output of these commands - do they look ok?
   git status
   git log     # press q to exit

And, if needed,

- :ref:`update the adastra_warpx.profile file <building-adastra-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-adastra-preparation>`.

As a last step, clean the build directory ``rm -rf $HOME/src/warpx/build_adastra`` and rebuild WarpX.


.. _running-cpp-adastra:

Running
-------

.. _running-cpp-adastra-MI250X-GPUs:

MI250X GPUs (2x64 GB)
^^^^^^^^^^^^^^^^^^^^^

In non-interactive runs:

.. literalinclude:: ../../../../Tools/machines/adastra-cines/submit.sh
   :language: bash
   :caption: You can copy this file from ``$HOME/src/warpx/Tools/machines/adastra-cines/submit.sh``.


.. _post-processing-adastra:

Post-Processing
---------------

.. note::

   TODO: Document any Jupyter or data services.

.. _known-adastra-issues:

Known System Issues
-------------------

.. warning::

   May 16th, 2022:
   There is a caching bug in Libfabric that causes WarpX simulations to occasionally hang on on more than 1 node.

   As a work-around, please export the following environment variable in your job scripts until the issue is fixed:

   .. code-block:: bash

      #export FI_MR_CACHE_MAX_COUNT=0  # libfabric disable caching
      # or, less invasive:
      export FI_MR_CACHE_MONITOR=memhooks  # alternative cache monitor

.. warning::

   Sep 2nd, 2022:
   rocFFT in ROCm 5.1-5.3 tries to `write to a cache <https://rocfft.readthedocs.io/en/latest/#runtime-compilation>`__ in the home area by default.
   This does not scale, disable it via:

   .. code-block:: bash

      export ROCFFT_RTC_CACHE_PATH=/dev/null

.. warning::

   January, 2023:
   We discovered a regression in AMD ROCm, leading to 2x slower current deposition (and other slowdowns) in ROCm 5.3 and 5.4.
   Reported to AMD and fixed for the next release of ROCm.

   Stay with the ROCm 5.2 module to avoid.
