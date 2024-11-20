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
Only very little software is pre-installed on LXPLUS so we show how to install from scratch all the dependencies using `Spack <https://spack.io>`__.

For size reasons it is not advisable to install WarpX in the ``$HOME`` directory, while it should be installed in the "work directory". For this purpose we set an environment variable with the path to the "work directory"

.. code-block:: bash

    export WORK=/afs/cern.ch/work/${USER:0:1}/$USER/

We clone WarpX in ``$WORK``:

.. code-block:: bash

    cd $WORK
    git clone https://github.com/ECP-WarpX/WarpX.git warpx

Installation profile file
^^^^^^^^^^^^^^^^^^^^^^^^^
The easiest way to install the dependencies is to use the pre-prepared ``warpx.profile`` as follows:

.. code-block:: bash

    cp $WORK/warpx/Tools/machines/lxplus-cern/lxplus_warpx.profile.example $WORK/lxplus_warpx.profile
    source $WORK/lxplus_warpx.profile

When doing this one can directly skip to the :ref:`Building WarpX <building-lxplus-warpx>` section.

To have the environment activated at every login it is then possible to add the following lines to the ``.bashrc``

.. code-block:: bash

    export WORK=/afs/cern.ch/work/${USER:0:1}/$USER/
    source $WORK/lxplus_warpx.profile

GCC
^^^
The pre-installed GNU compiler is outdated so we need a more recent compiler. Here we use the gcc 11.2.0 from the LCG project, but other options are possible.

We activate it by doing

.. code-block:: bash

    source /cvmfs/sft.cern.ch/lcg/releases/gcc/11.2.0-ad950/x86_64-centos7/setup.sh

In order to avoid using different compilers this line could be added directly into the ``$HOME/.bashrc`` file.

Spack
^^^^^
We download and activate Spack in ``$WORK``:

.. code-block:: bash

    cd $WORK
    git clone -c feature.manyFiles=true https://github.com/spack/spack.git
    source spack/share/spack/setup-env.sh

Now we add our gcc 11.2.0 compiler to spack:

.. code-block:: bash

    spack compiler find /cvmfs/sft.cern.ch/lcg/releases/gcc/11.2.0-ad950/x86_64-centos7/bin

Installing the Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the dependencies we create a virtual environment, which we call ``warpx-lxplus``:

.. code-block:: bash

    spack env create warpx-lxplus $WORK/WarpX/Tools/machines/lxplus-cern/spack.yaml
    spack env activate warpx-lxplus
    spack install

If the GPU support or the Python bindings are not needed, it's possible to skip the installation by respectively setting
the following environment variables export ``SPACK_STACK_USE_PYTHON=0`` and ``export SPACK_STACK_USE_CUDA = 0`` before
running the previous commands.

After the installation is done once, all we need to do in future sessions is just ``activate`` the environment again:

.. code-block:: bash

    spack env activate warpx-lxplus

The environment ``warpx-lxplus`` (or ``-cuda`` or ``-cuda-py``) must be reactivated everytime that we log in so it could
be a good idea to add the following lines to the ``.bashrc``:

.. code-block:: bash

    source $WORK/spack/share/spack/setup-env.sh
    spack env activate -d warpx-lxplus
    cd $HOME

.. _building-lxplus-warpx:

Building WarpX
^^^^^^^^^^^^^^

We prepare and load the Spack software environment as above.
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
   python3 -m pip install -U build packaging setuptools wheel
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
