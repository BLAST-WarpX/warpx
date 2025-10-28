.. _building-lxplus:

LXPLUS (CERN)
=============

The LXPLUS cluster is located at CERN.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `Lxplus documentation <https://lxplusdoc.web.cern.ch>`__
* Batch system: `HTCondor <https://batchdocs.web.cern.ch/index.html>`__
* Filesystem locations:
    * User folder: ``/afs/cern.ch/user/<a>/<account>`` (10GByte)
    * Work folder: ``/afs/cern.ch/work/<a>/<account>`` (100GByte)
    * Eos storage: ``/eos/home-<a>/<account>`` (1T)

Through LXPLUS we have access to CPU and GPU nodes (the latter equipped with NVIDIA V100 and T4 GPUs).


Installation
------------
For size reasons it is not advisable to install WarpX in the ``$HOME`` directory, while it should be installed in the "work directory". For this purpose we set an environment variable with the path to the "work directory"

.. code-block:: bash

    export WORK=/afs/cern.ch/work/${USER:0:1}/$USER/

We clone WarpX in ``$WORK``:

.. code-block:: bash

    cd $WORK
    git clone https://github.com/BLAST-WarpX/warpx.git warpx

Installation profile file
^^^^^^^^^^^^^^^^^^^^^^^^^
For convenience, all variables, cloning WarpX, and loading the LCG view required for the dependencies are available in the profile file ``warpx.profile`` as follows:

.. code-block:: bash

    cp $WORK/warpx/Tools/machines/lxplus-cern/lxplus_warpx.profile.example $WORK/lxplus_warpx.profile
    source $WORK/lxplus_warpx.profile

To have the environment activated at every login it is then possible to add the following lines to the ``.bashrc``

.. code-block:: bash

    export WORK=/afs/cern.ch/work/${USER:0:1}/$USER/
    source $WORK/lxplus_warpx.profile

Building WarpX
^^^^^^^^^^^^^^

All dependencies are available via the LCG software stack. We choose the CUDA software stack to be able to compile both with and without CUDA without changing the stack. As both a serial and a parallel HDF5 installation are available, one has to make sure that WarpX picks up the parallel one both at build and at run time. Therefore, we load the software stack and source an environment script that ensures that ``hdf5_mpi`` appears before the serial version:

.. code-block:: bash

    source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
    source /cvmfs/sft.cern.ch/lcg/releases/LCG_108_cuda/hdf5_mpi/1.14.6/x86_64-el9-gcc13-opt/hdf5_mpi-env.sh


Then we build WarpX:

.. code-block:: bash

    cmake -S . -B build -DWarpX_DIMS="1;2;RZ;3"
    cmake --build build -j 6

Or if we need to compile with CUDA:

.. code-block:: bash

    cmake -S . -B build -DWarpX_COMPUTE=CUDA -DWarpX_DIMS="1;2;RZ;3"
    cmake --build build -j 6

**That's it!**
A 3D WarpX executable is now in ``build/bin/`` and can be run with a :ref:`3D example inputs file <usage-examples>`.
Most people execute the binary directly or copy it out to a location in ``$WORK``.

Python Bindings
^^^^^^^^^^^^^^^

Here we assume that a Python interpreter has been set up as explained previously.

Now, ensure Python tooling is up-to-date:

.. code-block:: bash

   python3 -m pip install -U pip
   python3 -m pip install -U build packaging setuptools[core] wheel
   python3 -m pip install -U cmake

Then we compile WarpX as in the previous section (with or without CUDA) adding ``-DWarpX_PYTHON=ON`` and then we install it into our Python:

.. code-block:: bash

   cmake -S . -B build -DWarpX_COMPUTE=CUDA -DWarpX_DIMS="1;2;RZ;3" -DWarpX_APP=OFF -DWarpX_PYTHON=ON
   cmake --build build --target pip_install -j 6

This builds WarpX for 3D geometry.

Alternatively, if you like to build WarpX for all geometries at once, use:

.. code-block:: bash

   BUILD_PARALLEL=6 python3 -m pip wheel .
   python3 -m pip install pywarpx-*whl
