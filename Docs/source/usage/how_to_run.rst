.. _usage_run:

Run WarpX
=========

To run a new simulation, please follow these steps:

#. Create a **new directory**, where the simulation will run.
#. Make sure the WarpX **executable** is either copied into this directory or in your ``PATH`` `environment variable <https://en.wikipedia.org/wiki/PATH_(variable)>`__.
#. Add an **inputs file** in the same directory. On :ref:`HPC systems <install-hpc>`, add also a **job submission script**.
#. Run the executable.

Simulation Directory
--------------------

On Linux/macOS, this is as easy as:

.. code-block:: bash

   mkdir -p <run_directory>

Where ``<run_directory>`` is the actual path to the run directory.

Executable File
---------------

If you installed WarpX with a :ref:`package manager <install-methods>`, a ``warpx``-prefixed executable will be available as a regular system command.
Depending on build options, the executable name includes additional suffixes.
Try it like this:

.. code-block:: bash

   warpx<TAB>

Pressing the ``<TAB>`` key will suggest available WarpX executables found in your ``PATH`` `environment variable <https://en.wikipedia.org/wiki/PATH_(variable)>`__.

.. note::

      WarpX provides a separate binary for each dimensionality: 1D, 2D, 3D, RZ, RCYLINDER, and RSPHERE.
      We encode the supported dimensionality in the binary file name.

If you :ref:`compiled the code yourself <install-build-cmake>`, the WarpX executable is located in the source tree under ``build/bin``.
A symbolic link named ``warpx`` pointing to the most recently built executable is also created; you can copy either that link or the binary into your run directory.
Copy the **executable** to this directory:

.. code-block:: bash

   cp build/bin/<warpx_executable> <run_directory>/

where ``<warpx_executable>`` should be replaced by the actual name of the executable (see above) and ``<run_directory>`` by the actual path to the run directory.

Input File
----------

You need to provide WarpX with an input file that configures the simulation.
This can either be a parameter list or a Python script, depending on how you wish to run WarpX.

To run the WarpX executable, add a **parameter list** file in the directory (see :ref:`examples <usage-examples>` and :ref:`parameters <running-cpp-parameters>`).
This is a text file containing the numerical and physical parameters that define the simulation.

To run WarpX through the Python interface, add a **PICMI Python script** (see :ref:`examples <usage-examples>` and :ref:`PICMI parameters <usage-picmi-parameters>`).
This is a Python script that defines the numerical and physical parameters using the `PICMI standard <https://picmi-standard.org/>`__.

On :ref:`HPC systems <install-hpc>`, also copy and adjust a submission script that allocates computing nodes for you.
Please :ref:`reach out to us <contact>` if you need help setting up a template that runs with ideal performance.

Run Simulation
--------------

.. tab-set::

   .. tab-item:: WarpX Executable

      Run the executable directly, e.g. with MPI:

      .. code-block:: bash

         cd <run_directory>

         # run with an inputs file:
         mpirun -np <n_ranks> ./warpx <input_file>

      Here, ``<n_ranks>`` is the number of MPI ranks used, and ``<input_file>`` is the name of the parameter list.
      Note that the actual executable might have a longer name, depending on build options.

      The example above uses the copied executable in the current directory (``./``). If you installed WarpX with a package manager, omit the ``./`` because WarpX will be found in your ``PATH``.


   .. tab-item:: Python Script

      Run via the Python interface:

      .. code-block:: bash

         # run with a PICMI input script:
         mpirun -np <n_ranks> python <python_script>

      Here, ``<n_ranks>`` is the number of MPI ranks used, ``<python_script>`` is the name of the :ref:`PICMI <usage-picmi>` script.


   .. tab-item:: Job Script

      On an :ref:`HPC system <install-hpc>`, you would instead submit the :ref:`job script <install-hpc>` at this point, e.g. ``sbatch <submission_script>`` (SLURM) or ``bsub <submission_script>`` (LSF).


      .. tip::

         In the :ref:`next sections <running-cpp-parameters>`, we explain the parameters in the ``<input_file>``.
         You can also overwrite parameters from the command line, for example:

         .. code-block:: bash

            mpirun -np 4 ./warpx <input_file> max_step=10 warpx.numprocs=1 2 2


Outputs and Diagnostics
-----------------------

By default, WarpX writes status updates to the terminal (``stdout``).
On :ref:`HPC systems <install-hpc>`, it is common to store a copy of this in a file called ``outputs.txt``.

By default, WarpX also writes an exact copy of all explicitly and implicitly used input parameters to a file named ``warpx_used_inputs`` (this filename can be changed).
This is important for reproducibility, since, as noted above, options from the input file can be extended or overridden from the command line.

:ref:`Further configured diagnostics <running-cpp-parameters-diagnostics>` are explained in the next sections.
By default, they are written to a subdirectory in ``diags/`` and can use various :ref:`output formats <dataanalysis-formats>`.
