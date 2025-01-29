.. _building-fugaku:

Fugaku (Riken)
==============

The `Fugaku cluster <https://docs.nersc.gov/systems/perlmutter/>`_ is located at the Riken Center for Computational Science (Japan).


Introduction
------------

If you are new to this system, **please see the following resources**:

* `Fugaku user guide <https://www.r-ccs.riken.jp/en/fugaku/user-guide/>`__


.. _building-fugaku-preparation:

Preparation
-----------

Use the following commands to download the WarpX source code and switch to the correct branch:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx


Compiling WarpX on Fugaku is more practical on a compute node. Use the following commands to acquire a compute node for one hour:

.. code-block:: bash

   pjsub --interact -L "elapse=02:00:00" -L "node=1" --sparam "wait-time=300" --mpi "max-proc-per-node=48" --all-mount-gfscache


We use system software modules, add environment hints and further dependencies via the file ``$HOME/fugaku_warpx.profile``.
Create it now, modify it if needed, and source it (it will take few minutes):

.. code-block:: bash

   cp $HOME/src/warpx/Tools/machines/fugaku-riken/fugaku_warpx.profile.example $HOME/fugaku_warpx.profile
   source $HOME/fugaku_warpx.profile

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/fugaku-riken/fugaku_warpx.profile.example
      :language: bash

Finally, since Fugaku does not yet provide software modules for some of our dependencies, install them once:

.. code-block:: bash

   bash $HOME/src/warpx/Tools/machines/fugaku-riken/install_dependencies.sh

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../../Tools/machines/fugaku-riken/install_dependencies.sh
      :language: bash

.. _building-fugaku-compilation:

Compilation
-----------

Use the following :ref:`cmake commands <building-cmake>` to compile the application executable:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build

   export CC=$(which mpifcc)
   export CXX=$(which mpiFCC)
   export CFLAGS="-Nclang"
   export CXXFLAGS="-Nclang"

   cmake -S . -B build -DWarpX_COMPUTE=OMP \
       -DWarpX_DIMS="1;2;3" \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_CXX_FLAGS_RELEASE="-Ofast" \
       -DAMReX_DIFFERENT_COMPILER=ON \
       -DWarpX_MPI_THREAD_MULTIPLE=OFF

   cmake --build build -j 48

**That's it!**
A 3D WarpX executable is now in ``build/bin/`` and :ref:`can be run <running-cpp-fugaku>` with a :ref:`3D example inputs file <usage-examples>`.

.. _running-cpp-fugaku:

Running
-------

.. _running-cpp-fugaku-A64FX-CPUs:

A64FX CPUs
^^^^^^^^^^

In non-interactive runs, you can use `pjsub submit.sh` where `submit.sh` can be adapted from:

.. literalinclude:: ../../../../Tools/machines/fugaku-riken/submit.sh
   :language: bash
   :caption: You can copy this file from ``Tools/machines/fugaku-riken/submit.sh``.

Note: the ``Boost Eco Mode`` mode that is set in this example increases the default frequency of the A64FX
from 2 GHz to 2.2 GHz, while at the same time switching off one of the two floating-point arithmetic
pipelines. Some preliminary tests with WarpX show that this mode achieves performances similar to those of
the normal mode but with a reduction of the energy consumption of approximately 20%.
