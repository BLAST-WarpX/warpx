.. _install-methods:

Installation Methods
====================


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

Our community is here to help -- please `report installation issues <https://github.com/BLAST-WarpX/warpx/issues/new>`__ if you encounter any.

Please choose **one** of the installation methods below to get started.


.. only:: html

   .. image:: hpc.svg

HPC Systems
-----------

If you want to use WarpX on a specific high-performance computing (HPC) system, please go directly to our :ref:`HPC system-specific documentation <install-hpc>`.


.. _install-methods-conda:

.. only:: html

   .. image:: conda.svg

Using the conda-forge Package
-----------------------------

A package for WarpX is available via `conda-forge <https://conda-forge.org/download/>`__.

.. tip::

   We recommend disabling conda's automatic activation of the ``base`` environment.
   This helps `avoid interference with the system and other package managers <https://collegeville.github.io/CW20/WorkshopResources/WhitePapers/huebl-working-with-multiple-pkg-mgrs.pdf>`__.

   .. code-block:: bash

      conda config --set auto_activate_base false

   To ensure conda uses ``conda-forge`` as the only channel (which helps avoid issues with blocked ``defaults`` or ``anaconda`` repositories), set the following configuration options:

   .. code-block:: bash

      conda config --add channels conda-forge
      conda config --set channel_priority strict

.. code-block:: bash

   mamba create -n warpx -c conda-forge warpx
   mamba activate warpx

.. note::

   The ``warpx`` package on conda-forge does not yet provide `GPU support <https://github.com/conda-forge/warpx-feedstock/issues/89>`__.

.. _install-methods-spack:

.. only:: html

   .. image:: spack.svg

Using the Spack Package
-----------------------

Packages for WarpX are available via the `Spack <https://spack.readthedocs.io>`__ package manager.
The ``warpx`` package installs executables. The ``warpx +python`` variant also builds Python bindings, which can be used with `PICMI <https://github.com/picmi-standard/picmi>`__.

.. code-block:: bash

   # optional: activate Spack binary caches
   spack mirror add rolling https://binaries.spack.io/develop
   spack buildcache keys --install --trust

   # see `spack info py-warpx` for build options.
   # optional arguments:       -mpi compute=cuda
   spack install warpx +python
   spack load warpx +python

See ``spack info warpx`` and `the official Spack tutorial <https://spack-tutorial.readthedocs.io>`__ for more information.

.. _install-methods-pypi:

.. only:: html

   .. image:: pypi.svg

Using the PyPI Package
----------------------

If you have the :ref:`WarpX dependencies <install-dependencies>` installed, you can use ``pip`` to install WarpX (with PICMI) :ref:`from source <install-build-cmake>`:

.. code-block:: bash

   python3 -m pip install -U pip
   python3 -m pip install -U build packaging setuptools[core] wheel
   python3 -m pip install -U cmake

   python3 -m pip wheel -v git+https://github.com/BLAST-WarpX/warpx.git
   python3 -m pip install *whl

Pre-compiled binary packages will be published on `PyPI <https://pypi.org/>`__ in the future for faster installs.
Please consider using :ref:`conda <install-methods-conda>` in the meantime.

.. _install-methods-brew:

.. only:: html

   .. image:: brew.svg

Using the Brew Package
----------------------

.. note::

   Coming soon.


.. _install-methods-cmake:

.. only:: html

   .. image:: cmake.svg

From Source with CMake
----------------------

After installing the :ref:`WarpX dependencies <install-dependencies>`, you can also install WarpX from source with `CMake <https://cmake.org/>`_:

.. code-block:: bash

   # get the source code
   git clone https://github.com/BLAST-WarpX/warpx.git $HOME/src/warpx
   cd $HOME/src/warpx

   # configure
   cmake -S . -B build

   # optional: change configuration
   ccmake build

   # compile
   #   on Windows:          --config RelWithDebInfo
   cmake --build build -j 4

   # executables for WarpX are now in build/bin/

For more details on how to configure WarpX from source, please see the section :ref:`Build from Source <install-build-cmake>`.

.. _install-tips-macos:

Tips for macOS Users
--------------------

.. tip::

   Before using package managers, check for manually installed software under ``/usr/local``.
   If you find entries in ``bin/``, ``lib/``, etc., that look like MPI, HDF5, or other previously installed software, remove them first.

   If you find software such as MPI in those directories as symbolic links, it likely means you previously installed them with `brew <https://brew.sh/>`__.
   If you plan to use a package manager other than ``brew``, first run `brew unlink ... <https://docs.brew.sh/Tips-N%27-Tricks#quickly-remove-something-from-usrlocal>`__ on those packages to avoid incompatibilities.

See also: A. Huebl, `Working With Multiple Package Managers <https://collegeville.github.io/CW20/WorkshopResources/WhitePapers/huebl-working-with-multiple-pkg-mgrs.pdf>`__, `Collegeville Workshop (CW20) <https://collegeville.github.io/CW20/>`_, 2020
