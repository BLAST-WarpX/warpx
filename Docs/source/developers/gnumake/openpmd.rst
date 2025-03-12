.. _building-openpmd:

Building WarpX with support for openPMD output
==============================================

WarpX can dump data in the `openPMD format <https://github.com/openPMD>`__.
This feature currently requires to have a parallel version of HDF5 installed ;
therefore we recommend to use `spack <https://
spack.io>`__ in order to facilitate the installation.

More specifically, we recommend that you try installing the
`openPMD-api library 0.16.1 or newer <https://openpmd-api.readthedocs.io/en/0.16.1/>`__
using spack (first section below). If this fails, a back-up solution
is to install parallel HDF5 with spack, and then install the openPMD-api
library from source.

In order to install spack, you can simply do:

.. code-block:: bash

   git clone https://github.com/spack/spack.git
   export SPACK_ROOT=$PWD/spack
   . $SPACK_ROOT/share/spack/setup-env.sh

You may want to auto-activate spack when you open a new terminal by adding this to your ``$HOME/.bashrc`` file:

.. code-block:: bash

   echo -e "# activate spack package manager\n. ${SPACK_ROOT}/share/spack/setup-env.sh" >> $HOME/.bashrc


.. _building-openpmd-spack:

WarpX Development Environment with Spack
----------------------------------------

Create and activate a Spack environment with all software needed to build WarpX

.. code-block:: bash

   spack env create warpx-dev    # you do this once
   spack env activate warpx-dev
   spack add gmake
   spack add mpi
   spack add openpmd-api
   spack add pkg-config
   spack install

This will download and compile all dependencies.

Whenever you need this development environment in the future, just repeat the quick ``spack env activate warpx-dev`` step.
For example, we can now compile WarpX by ``cd``-ing into the ``WarpX`` folder and typing:

.. code-block:: bash

   spack env activate warpx-dev
   make -j 4 USE_OPENPMD=TRUE

You will also need to load the same spack environment when running WarpX, for instance:

.. code-block:: bash

   spack env activate warpx-dev
   mpirun -np 4 ./warpx.exe inputs

You can check which Spack environments exist and if one is still active with

.. code-block:: bash

   spack env list  # already created environments
   spack env st    # is an environment active?


.. _building-openpmd-source:

Installing openPMD-api from source
----------------------------------

You can also build openPMD-api from source, e.g. to build against the module environment of a supercomputer cluster.

First, load the according modules of the cluster to support the openPMD-api dependencies.
You can find the `required and optional dependencies here <https://github.com/openPMD/openPMD-api#dependencies>`__.

You usually just need a C++ compiler, CMake, and one or more file backend libraries, such as HDF5 and/or ADIOS2.

If optional dependencies are installed in non-system paths, one needs to `hint their installation location <https://hsf-training.github.io/hsf-training-cmake-webpage/09-findingpackages/index.html>`_ with an environment variable during the build phase:

.. code-block:: bash

   # optional: only if you manually installed HDF5 and/or ADIOS2 in custom directories
   export HDF5_ROOT=$HOME/path_to_installed_software/hdf5-1.12.0/
   export ADIOS2_ROOT=$HOME/path_to_installed_software/adios2-2.7.1/

Then, in the ``$HOME/warpx_directory/``, download and build openPMD-api:

.. code-block:: bash

   git clone https://github.com/openPMD/openPMD-api.git
   mkdir openPMD-api-build
   cd openPMD-api-build
   cmake ../openPMD-api -DopenPMD_USE_PYTHON=OFF -DCMAKE_INSTALL_PREFIX=$HOME/warpx_directory/openPMD-install/ -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_RPATH='$ORIGIN'
   cmake --build . --target install

Finally, compile WarpX:

.. code-block:: bash

   cd ../WarpX
   # Note that one some systems, /lib might need to be replaced with /lib64.
   export PKG_CONFIG_PATH=$HOME/warpx_directory/openPMD-install/lib/pkgconfig:$PKG_CONFIG_PATH
   export CMAKE_PREFIX_PATH=$HOME/warpx_directory/openPMD-install:$CMAKE_PREFIX_PATH

   make -j 4 USE_OPENPMD=TRUE

.. note::

   If you compile with :ref:`CMake <install-developers>`, all you need to add is the ``-DWarpX_OPENPMD=ON`` option (on by default), and we will download and build openPMD-api on-the-fly.

When running WarpX, we will recall where you installed openPMD-api via RPATHs, so you just need to load the same module environment as used for building (same MPI, HDF5, ADIOS2, for instance).

.. code-block:: bash

   # module load ...  (compiler, MPI, HDF5, ADIOS2, ...)

   mpirun -np 4 ./warpx.exe inputs
