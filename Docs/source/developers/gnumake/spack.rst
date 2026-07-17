.. _developers-gnumake-spack:

Building WarpX with Spack
=========================

As mentioned in the :ref:`install section <install-methods>`, WarpX can be installed using Spack.
From the `Spack web page <https://spack.io>`__: "Spack is a package management tool designed to support multiple versions and configurations of software on a wide variety of platforms and environments."

.. note::

   Quick-start hint for macOS users:
   Before getting started with Spack, please check what you manually installed in ``/usr/local``.
   If you find entries in ``bin/``, ``lib/`` et al. that look like you manually installed MPI, HDF5 or other software at some point, then remove those files first.

   If you find software such as MPI in the same directories that are shown as symbolic links then it is likely you `brew installed <https://brew.sh>`_ software before.
   Run `brew unlink ... <https://docs.brew.sh/Tips-N%27-Tricks#quickly-remove-something-from-usrlocal>`_ on such packages first to avoid software incompatibilities.

Spack is available from `github <https://github.com/spack/spack>`_.
Spack only needs to be cloned and can be used right away - there are no installation steps.
You can add `binary caches <https://spack.io/spack-binary-packages/>`__ for faster builds:

.. code-block:: bash

   spack mirror add rolling https://binaries.spack.io/develop
   spack buildcache keys --install --trust

Do not miss out on `the official Spack tutorial <https://spack-tutorial.readthedocs.io/>`_ if you are new to Spack.

The spack command, ``spack/bin/spack``, can be used directly or ``spack/bin`` can be added to your ``PATH`` environment variable.

WarpX is built with the single command

.. code-block:: bash

   spack install warpx

This will build the 3-D version of WarpX using the ``development`` branch.
At the very end of the output from build sequence, Spack tells you where the WarpX executable has been placed.
Alternatively, ``spack load warpx`` can be called, which will put the executable in your ``PATH`` environment variable.

WarpX can be built in several variants, see

.. code-block:: bash

   spack info warpx
   spack info py-warpx

for all available options.

For example

.. code-block:: bash

   spack install warpx dims=2 build_type=Debug

will build the 2-D version and also turns debugging on.

See ``spack help --spec`` for all syntax details.
Also, please consult the `basic usage section <https://spack.readthedocs.io/en/latest/basic_usage.html>`_ of the Spack package manager for an extended introduction to Spack.

The Python version of WarpX is available through the ``py-warpx`` package.
