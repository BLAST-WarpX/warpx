.. _developers-gnumake-python:

Installing WarpX as a Python package
====================================

A full Python installation of WarpX can be done, which includes a build of all of the C++ code, or a pure Python version can be made which only installs the Python scripts. WarpX requires Python version 3.9 or newer.

For a full Python installation of WarpX
---------------------------------------

WarpX' Python bindings depend on ``numpy``, ``periodictable``, ``picmistandard``, and ``mpi4py``.

Type

.. code-block:: bash

   make -j 4 USE_PYTHON_MAIN=TRUE

or edit the ``GNUmakefile`` and set ``USE_PYTHON_MAIN=TRUE``, and type

.. code-block:: bash

   make -j 4

Additional compile time options can be specified as needed.
This will compile the code, and install the Python bindings and the Python scripts as a package (named ``pywarpx``) in your standard Python installation (i.e. in your ``site-packages`` directory).

If you do not have write permission to the default Python installation (e.g. typical on computer clusters), there are two options.
The recommended option is to use a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`__, which provides the most flexibility and robustness.

Alternatively, add the ``--user`` install option to have WarpX installed elsewhere.

.. code-block:: bash

   make -j 4 PYINSTALLOPTIONS=--user

With ``--user``, the default location will be in your home directory, ``~/.local``, or the location defined by the environment variable ``PYTHONUSERBASE``.

In HPC environments, it is often recommended to install codes in scratch or work space which typically have faster disk access.

The different dimensioned versions of WarpX, 3D, 2D, and RZ, can coexist in the Python installation.
The appropriate one will be imported depending on the input file.
Note, however, other options will overwrite - compiling with ``DEBUG=TRUE`` will replace the version compiled with ``DEBUG=FALSE`` for example.


For a pure Python installation
------------------------------

This avoids the compilation of the C++ and is recommended when only using the Python input files as preprocessors.
This installation depend on ``numpy``, ``periodictable``, and ``picmistandard``.

Go into the ``Python`` subdirectory and run

.. code-block:: bash

   python setup.py install

This installs the Python scripts as a package (named ``pywarpx``) in your standard Python installation (i.e. in your ``site-packages`` directory).
If you do not have write permission to the default Python installation (e.g. typical on computer clusters), there are two options.
The recommended option is to use a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`__, which provides the most flexibility and robustness.

Alternatively, add the ``--user`` install option to have WarpX installed elsewhere.

.. code-block:: bash

   python setup.py install --user

With ``--user``, the default location will be in your home directory, ``~/.local``, or the location defined by the environment variable ``PYTHONUSERBASE``.
