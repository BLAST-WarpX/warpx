.. _building-leonardo:

Leonardo (CINECA)
=================

The `Leonardo cluster <https://leonardo-supercomputer.cineca.eu/>`_ is hosted at `CINECA <https://www.cineca.it/en>`_.

On Leonardo, each one of the 3456 compute nodes features a custom Atos Bull Sequana XH21355 "Da Vinci" blade, composed of:

* 1 x CPU Intel Ice Lake Xeon 8358 32 cores 2.60 GHz
* 512 (8 x 64) GB RAM DDR4 3200 MHz
* 4 x NVidia custom Ampere A100 GPU 64GB HBM2
* 2 x NVidia HDR 2×100 GB/s cards

Introduction
------------

If you are new to this system, **please see the following resources**:

* `Leonardo website <https://leonardo-supercomputer.cineca.eu/>`_
* `Leonardo user guide <https://wiki.u-gov.it/confluence/display/SCAIUS/UG3.2%3A+LEONARDO+UserGuide>`_

Storage organization:

* ``$HOME``: permanent, backed up, user specific (50 GB quota)
* ``$CINECA_SCRATCH``: temporary, user specific, no backup, a large disk for the storage of run time data and files, automatic cleaning procedure of data older than 40 days
* ``$PUBLIC``: permanent, no backup (50 GB quota)
* ``$WORK``: permanent, project specific, no backup

.. _building-leonardo-preparation:

Preparation
-----------

Use the following commands to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx

We use system software modules, add environment hints and further dependencies via the file ``$HOME/leonardo_gpu_warpx.profile``.
Create it now:

.. code-block:: bash

   cp $HOME/src/warpx/Tools/machines/leonardo-cineca/leonardo_gpu_warpx.profile.example $HOME/leonardo_gpu_warpx.profile

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/leonardo-cineca/leonardo_gpu_warpx.profile.example
      :language: bash

.. important::

   Now, and as the first step on future logins to Leonardo, activate these environment settings:

   .. code-block:: bash

      source $HOME/leonardo_gpu_warpx.profile

Finally, since Leonardo does not yet provide software modules for some of our dependencies, install them once:

.. code-block:: bash

   bash $HOME/src/warpx/Tools/machines/leonardo-cineca/install_gpu_dependencies.sh
   source $HOME/sw/venvs/warpx/bin/activate

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/leonardo-cineca/install_gpu_dependencies.sh
      :language: bash


.. _building-leonardo-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build_gpu

   cmake -S . -B build_gpu -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_gpu -j 16

The WarpX application executables are now in ``$HOME/src/warpx/build_gpu/bin/``.
Additionally, the following commands will install WarpX as a Python module:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build_gpu_py

   cmake -S . -B build_gpu_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_PYTHON=ON -DWarpX_APP=OFF -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_gpu_py -j 16 --target pip_install

Now, you can :ref:`submit Leonardo compute jobs <running-leonardo>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit Leonardo jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-leonardo>` or copy them to a location in ``$CINECA_SCRATCH``.

.. _building-leonardo-update:

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

- :ref:`update the leonardo_gpu_warpx.profile file <building-leonardo-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-leonardo-preparation>`.

As a last step, clean the build directories ``rm -rf $HOME/src/warpx/build_gpu*`` and rebuild WarpX.


.. _running-leonardo:

Running
-------
The batch script below can be used to run a WarpX simulation on multiple nodes on Leonardo.
Replace descriptions between chevrons ``<>`` by relevant values.
Note that we run one MPI rank per GPU.

.. literalinclude:: ../../../../Tools/machines/leonardo-cineca/job.sh
   :language: bash
   :caption: You can copy this file from ``$HOME/src/warpx/Tools/machines/leonardo-cineca/job.sh``.

To run a simulation, copy the lines above to a file ``job.sh`` and run

.. code-block:: bash

   sbatch job.sh

to submit the job.

.. _post-processing-leonardo:

Post-Processing
---------------

For post-processing, activate the environment settings:

.. code-block:: bash

  source $HOME/leonardo_gpu_warpx.profile

and run python scripts.
