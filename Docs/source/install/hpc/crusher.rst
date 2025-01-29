.. _building-crusher:

Crusher (OLCF)
==============

The `Crusher cluster <https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html>`_ is located at OLCF.

On Crusher, each compute node provides four AMD MI250X GPUs, each with two Graphics Compute Dies (GCDs) for a total of 8 GCDs per node.
You can think of the 8 GCDs as 8 separate GPUs, each having 64 GB of high-bandwidth memory (HBM2E).


Introduction
------------

If you are new to this system, **please see the following resources**:

* `Crusher user guide <https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html>`_
* Batch system: `Slurm <https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html#running-jobs>`_
* `Production directories <https://docs.olcf.ornl.gov/data/index.html#data-storage-and-transfers>`_:

  * ``$HOME``: per-user directory, use only for inputs, source and scripts; backed up; mounted as read-only on compute nodes, that means you cannot run in it (50 GB quota)
  * ``$PROJWORK/$proj/``: shared with all members of a project, purged every 90 days, Lustre (recommended)
  * ``$MEMBERWORK/$proj/``: single user, purged every 90 days, Lustre (usually smaller quota, 50TB default quota)
  * ``$WORLDWORK/$proj/``: shared with all users, purged every 90 days, Lustre (50TB default quota)

Note: the Orion Lustre filesystem on Frontier and Crusher, and the older Alpine GPFS filesystem on Summit are not mounted on each others machines.
Use `Globus <https://www.globus.org>`__ to transfer data between them if needed.


.. _building-crusher-preparation:

Preparation
-----------

Use the following commands to download the WarpX source code:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx

We use system software modules, add environment hints and further dependencies via the file ``$HOME/crusher_warpx.profile``.
Create it now:

.. code-block:: bash

   cp $HOME/src/warpx/Tools/machines/crusher-olcf/crusher_warpx.profile.example $HOME/crusher_warpx.profile

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/crusher-olcf/crusher_warpx.profile.example
      :language: bash

Edit the 2nd line of this script, which sets the ``export proj=""`` variable.
For example, if you are member of the project ``aph114``, then run ``vi $HOME/crusher_warpx.profile``.
Enter the edit mode by typing ``i`` and edit line 2 to read:

.. code-block:: bash

   export proj="aph114"

Exit the ``vi`` editor with ``Esc`` and then type ``:wq`` (write & quit).

.. important::

   Now, and as the first step on future logins to Crusher, activate these environment settings:

   .. code-block:: bash

      source $HOME/crusher_warpx.profile

Finally, since Crusher does not yet provide software modules for some of our dependencies, install them once:

.. code-block:: bash

   bash $HOME/src/warpx/Tools/machines/crusher-olcf/install_dependencies.sh
   source $HOME/sw/crusher/gpu/venvs/warpx-crusher/bin/activate

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/crusher-olcf/install_dependencies.sh
      :language: bash


.. _building-crusher-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build_crusher

   cmake -S . -B build_crusher -DWarpX_COMPUTE=HIP -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_crusher -j 16

The WarpX application executables are now in ``$HOME/src/warpx/build_crusher/bin/``.
Additionally, the following commands will install WarpX as a Python module:

.. code-block:: bash

   rm -rf build_crusher_py

   cmake -S . -B build_crusher_py -DWarpX_COMPUTE=HIP -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
   cmake --build build_crusher_py -j 16 --target pip_install

Now, you can :ref:`submit Crusher compute jobs <running-cpp-crusher>` for WarpX :ref:`Python (PICMI) scripts <usage-picmi>` (:ref:`example scripts <usage-examples>`).
Or, you can use the WarpX executables to submit Crusher jobs (:ref:`example inputs <usage-examples>`).
For executables, you can reference their location in your :ref:`job script <running-cpp-crusher>` or copy them to a location in ``$PROJWORK/$proj/``.


.. _building-crusher-update:

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

- :ref:`update the crusher_warpx.profile file <building-crusher-preparation>`,
- log out and into the system, activate the now updated environment profile as usual,
- :ref:`execute the dependency install scripts <building-crusher-preparation>`.

As a last step, clean the build directory ``rm -rf $HOME/src/warpx/build_crusher`` and rebuild WarpX.


.. _running-cpp-crusher:

Running
-------

.. _running-cpp-crusher-MI250X-GPUs:

MI250X GPUs (2x64 GB)
^^^^^^^^^^^^^^^^^^^^^

After requesting an interactive node with the ``getNode`` alias above, run a simulation like this, here using 8 MPI ranks and a single node:

.. code-block:: bash

   runNode ./warpx inputs

Or in non-interactive runs:

.. literalinclude:: ../../../../Tools/machines/crusher-olcf/submit.sh
   :language: bash
   :caption: You can copy this file from ``$HOME/src/warpx/Tools/machines/crusher-olcf/submit.sh``.


.. _post-processing-crusher:

Post-Processing
---------------

For post-processing, most users use Python via OLCFs's `Jupyter service <https://jupyter.olcf.ornl.gov>`__ (`Docs <https://docs.olcf.ornl.gov/services_and_applications/jupyter/index.html>`__).

Please follow the same guidance as for :ref:`OLCF Summit post-processing <post-processing-summit>`.

.. _known-crusher-issues:

Known System Issues
-------------------

.. note::

    Please see the :ref:`Frontier Known System Issues <known-frontier-issues>` due to the similarity of the two systems.
