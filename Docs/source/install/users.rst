.. _install-users:

Users
=====

.. raw:: html

   <style>
   .rst-content section>img {
       width: 30px;
       margin-bottom: 0;
       margin-top: 0;
       margin-right: 15px;
       margin-left: 15px;
       float: left;
   }
   </style>

Our community is here to help.
Please `report installation problems <https://github.com/ECP-WarpX/WarpX/issues/new>`_ in case you should get stuck.

Choose **one** of the installation methods below to get started:


.. only:: html

   .. image:: hpc.svg

HPC Systems
-----------

If want to use WarpX on a specific high-performance computing (HPC) systems, jump directly to our :ref:`HPC system-specific documentation <install-hpc>`.


.. _install-conda:

.. only:: html

   .. image:: conda.svg

Using the Conda-Forge Package
-----------------------------

A package for WarpX is available via `Conda-Forge <https://conda-forge.org/download/>`__.

.. tip::

   We recommend to deactivate that conda self-activates its ``base`` environment.
   This `avoids interference with the system and other package managers <https://collegeville.github.io/CW20/WorkshopResources/WhitePapers/huebl-working-with-multiple-pkg-mgrs.pdf>`__.

   .. code-block:: bash

      conda config --set auto_activate_base false

   In order to make sure that the conda configuration uses ``conda-forge`` as the only channel, which will help avoid issues with blocked ``defaults`` or ``anaconda`` repositories, please set the following configurations:

   .. code-block:: bash

      conda config --add channels conda-forge
      conda config --set channel_priority strict

.. code-block:: bash

   mamba create -n warpx -c conda-forge warpx
   mamba activate warpx

.. note::

   The ``warpx`` package on conda-forge does not yet provide `GPU support <https://github.com/conda-forge/warpx-feedstock/issues/89>`__.


.. _install-spack:

.. only:: html

   .. image:: spack.svg

Using the Spack Package
-----------------------

Packages for WarpX are available via the `Spack <https://spack.readthedocs.io>`__ package manager.
The package ``warpx`` installs executables and the package ``py-warpx`` includes Python bindings, i.e. `PICMI <https://github.com/picmi-standard/picmi>`_.

.. code-block:: bash

   # optional: activate Spack binary caches
   spack mirror add rolling https://binaries.spack.io/develop
   spack buildcache keys --install --trust

   # see `spack info py-warpx` for build options.
   # optional arguments:  -mpi ^warpx dims=2 compute=cuda
   spack install py-warpx
   spack load py-warpx

See ``spack info warpx`` or ``spack info py-warpx`` and `the official Spack tutorial <https://spack-tutorial.readthedocs.io>`__ for more information.


.. _install-pypi:

.. only:: html

   .. image:: pypi.svg

Using the PyPI Package
----------------------

Given that you have the :ref:`WarpX dependencies <install-dependencies>` installed, you can use ``pip`` to install WarpX with `PICMI <https://github.com/picmi-standard/picmi>`_ :ref:`from source <install-developers>`:

.. code-block:: bash

   python3 -m pip install -U pip
   python3 -m pip install -U build packaging setuptools wheel
   python3 -m pip install -U cmake

   python3 -m pip wheel -v git+https://github.com/ECP-WarpX/WarpX.git
   python3 -m pip install *whl

In the future, will publish pre-compiled binary packages on `PyPI <https://pypi.org/>`__ for faster installs.
(Consider using :ref:`conda <install-conda>` in the meantime.)


.. _install-brew:

.. only:: html

   .. image:: brew.svg

Using the Brew Package
----------------------

.. note::

   Coming soon.


.. _install-cmake:

.. only:: html

   .. image:: cmake.svg

From Source with CMake
----------------------

After installing the :ref:`WarpX dependencies <install-dependencies>`, you can also install WarpX from source with `CMake <https://cmake.org/>`_:

.. code-block:: bash

   # get the source code
   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx
   cd $HOME/src/warpx

   # configure
   cmake -S . -B build

   # optional: change configuration
   ccmake build

   # compile
   #   on Windows:          --config RelWithDebInfo
   cmake --build build -j 4

   # executables for WarpX are now in build/bin/

We document the details in the :ref:`developer installation <install-developers>`.


.. _install-users-macos:

Tips for macOS Users
--------------------

.. tip::

   Before getting started with package managers, please check what you manually installed in ``/usr/local``.
   If you find entries in ``bin/``, ``lib/`` et al. that look like you manually installed MPI, HDF5 or other software in the past, then remove those files first.

   If you find software such as MPI in the same directories that are shown as symbolic links then it is likely you `brew installed <https://brew.sh/>`__ software before.
   If you are trying annother package manager than ``brew``, run `brew unlink ... <https://docs.brew.sh/Tips-N%27-Tricks#quickly-remove-something-from-usrlocal>`__ on such packages first to avoid software incompatibilities.

See also: A. Huebl, `Working With Multiple Package Managers <https://collegeville.github.io/CW20/WorkshopResources/WhitePapers/huebl-working-with-multiple-pkg-mgrs.pdf>`__, `Collegeville Workshop (CW20) <https://collegeville.github.io/CW20/>`_, 2020
