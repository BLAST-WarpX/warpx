.. _building-dane:

Dane (LLNL)
=============

The `Dane Intel CPU cluster <https://hpc.llnl.gov/hardware/compute-platforms/dane>`__ is located at LLNL.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `LLNL user account <https://lc.llnl.gov>`__ (login required)
* `Jupyter service <https://lc.llnl.gov/jupyter>`__ (`documentation <https://lc.llnl.gov/confluence/display/LC/JupyterHub+and+Jupyter+Notebook>`__, login required)
* `Production directories <https://hpc.llnl.gov/hardware/file-systems>`__:

  * ``/p/lustre1/$(whoami)`` and ``/p/lustre2/$(whoami)``: personal directory on the parallel filesystem
  * Note that the ``$HOME`` directory and the ``/usr/workspace/$(whoami)`` space are NFS mounted and *not* suitable for production quality data generation.


.. _building-dane-preparation:

Preparation
-----------

Use the following commands to download the WarpX source code.
Note that these commands and the shell scripts all assume the bash shell.

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git /usr/workspace/${USER}/dane/src/warpx

We use system software modules, add environment hints and further dependencies via the file ``$HOME/dane_warpx.profile``.
Create it now:

.. code-block:: bash

   cp /usr/workspace/${USER}/dane/src/warpx/Tools/machines/dane-llnl/dane_warpx.profile.example $HOME/dane_warpx.profile

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/dane-llnl/dane_warpx.profile.example
      :language: bash

Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
For example, if you are member of the project ``tps``, then run ``vi $HOME/dane_warpx.profile``.
Enter the edit mode by typing ``i`` and edit line 2 to read:

.. code-block:: bash

   export proj="tps"

Exit the ``vi`` editor with ``Esc`` and then type ``:wq`` (write & quit).

.. important::

   Now, and as the first step on future logins to Dane, activate these environment settings:

   .. code-block:: bash

      source $HOME/dane_warpx.profile

Finally, since Dane does not yet provide software modules for some of our dependencies, install them once:

.. code-block:: bash

   bash /usr/workspace/${USER}/dane/src/warpx/Tools/machines/dane-llnl/install_dependencies.sh
   source /usr/workspace/${USER}/dane/venvs/warpx-dane/bin/activate

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/dane-llnl/install_dependencies.sh
      :language: bash


.. _building-dane-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. code-block:: bash

   cd /usr/workspace/${USER}/dane/src/warpx
   rm -rf build_dane

   cmake -S . -B build_dane -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_dane -j 6

The WarpX application executables are now in ``/usr/workspace/${USER}/dane/src/warpx/build_dane/bin/``.
Additionally, the following commands will install WarpX as a Python module:

.. code-block:: bash

   rm -rf build_dane_py

   cmake -S . -B build_dane_py -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_dane_py -j 6 --target pip_install

Now, you can :ref:`submit Dane compute jobs <running-cpp-dane>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit Dane jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-cpp-dane>` or copy them to a location in ``$PROJWORK/$proj/``.


.. _building-dane-update:

Update WarpX & Dependencies
---------------------------

If you already installed WarpX in the past and want to update it, start by getting the latest source code:

.. code-block:: bash

   cd /usr/workspace/${USER}/dane/src/warpx

   # read the output of this command - does it look ok?
   git status

   # get the latest WarpX source code
   git fetch
   git pull

   # read the output of these commands - do they look ok?
   git status
   git log     # press q to exit

And, if needed,

- :ref:`update the dane_warpx.profile file <building-dane-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-dane-preparation>`.

As a last step, clean the build directory ``rm -rf /usr/workspace/${USER}/dane/src/warpx/build_dane`` and rebuild WarpX.


.. _running-cpp-dane:

Running
-------

.. _running-cpp-dane-CPUs:

Intel Sapphire Rapids CPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^

The batch script below can be used to run a WarpX simulation on 2 nodes on the supercomputer Dane at LLNL.
Replace descriptions between chevrons ``<>`` by relevant values, for instance ``<input file>`` could be ``plasma_mirror_inputs``.

.. literalinclude:: ../../../../Tools/machines/dane-llnl/dane.sbatch
   :language: bash
   :caption: You can copy this file from ``Tools/machines/dane-llnl/dane.sbatch``.

To run a simulation, copy the lines above to a file ``dane.sbatch`` and run

.. code-block:: bash

   sbatch dane.sbatch

to submit the job.
