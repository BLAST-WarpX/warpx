.. _usage-python-extend:

Extend a Simulation with Python
===============================

Overview
--------

WarpX's Python bindings let you integrate Python code directly into a WarpX simulation.
Through this interface, you can **access and modify simulation data** -- such as particle properties, field values -- as the simulation runs.
This versatility opens the door to a wide range of workflows, including:

   - **Adding a custom physics module** (for instance, a specific collision model) that may not yet be available in WarpX's C++ implementation, and that can be quickly implemented in Python.
   - **Coupling WarpX with another simulation tool** that has a Python interface, enabling both codes to operate on the same particle or field data.
   - **Incorporating AI-based surrogate models** built in Python (e.g., with PyTorch or TensorFlow) to emulate complex physical processes.

If your custom Python code uses high-performance, GPU-accelerated libraries -- such as `cupy <https://cupy.dev/>`__, `pytorch <https://pytorch.org/>`__,
or `numba <https://numba.pydata.org/>`__ -- the extra computations are unlikely to significantly impact simulation speed.
Note that WarpX's Python bindings provide direct access to particle and field data without creating copies, resulting in very low overhead.

.. _usage-python-extend-run-simulation:

How to run a simulation with Python extensions
----------------------------------------------

- **Install WarpX with support for the Python interface**: for instance, if you :ref:`compile WarpX from source <install-build-code>`, this involves using ``-DWarpX_PYTHON=ON``.

- **Write a Python script that extends the simulation**: this can be done starting from a simulation defined either with a :ref:`parameter list <running-cpp-parameters>` or with the :ref:`PICMI Python interface <usage-picmi>`.
  The Python script typically contains :ref:`callback functions <usage-python-extend-callbacks>` that :ref:`access/modify <usage-python-extend-data-access>` the simulation data (see the sections below for more details).

.. tab-set::

   .. tab-item:: Parameter List

      When starting from a :ref:`parameter list <running-cpp-parameters>`, write a Python script that loads the parameter list file using the :py:meth:`~pywarpx.picmi.Simulation.load_inputs_file` method:

      .. code-block:: python3

         from pywarpx import warpx

         sim = warpx
         sim.load_inputs_file("./inputs_test_3d_laser_acceleration")

         # register callbacks ...

         # advance simulation until the last time step
         sim.step()

      .. dropdown:: Full Example

         .. literalinclude:: inputs_test_3d_laser_acceleration_python.py
            :language: python3
            :caption: You can copy this file from ``Examples/Physics_applications/laser_acceleration/inputs_test_3d_laser_acceleration_python.py`` and it requires the files ``inputs_test_3d_laser_acceleration`` and ``inputs_base_3d`` from the same folder.

   .. tab-item:: PICMI

      When starting from a :ref:`PICMI Python script <usage-picmi>`, simply add the Python code that extends the simulation to this script, before the call to :py:meth:`~pywarpx.picmi.Simulation.step`.

      .. code-block:: python3

         # Preparation: set up the simulation
         #   sim = picmi.Simulation(...)
         #   ...

         # register callbacks ...

         sim.step(nsteps=1000)


- **Then, run the simulation by executing the Python script**: for instance using ``mpirun`` or ``srun`` on an HPC system.

.. code-block:: bash

   mpirun -np <n_ranks> python <python_script>

.. _usage-python-extend-callbacks:

Callback Functions
------------------

Installing `callback functions <https://en.wikipedia.org/wiki/Callback_(computer_programming)>`__ will execute a given Python function at a
specific location in the WarpX simulation loop. The syntax to use in order to define callback functions is described in the links below.

.. toctree::
   :maxdepth: 1

   python_callbacks

.. _usage-python-extend-data-access:

Accessing simulation data through Python
----------------------------------------

While the simulation is running, the Python code (e.g. the code in the callback functions) will have read and write access the WarpX simulation data.
The specific Python syntax to access this data is described in the following sections.

.. toctree::
   :maxdepth: 1

   python_field_data
   python_particle_data
   python_particle_boundary_data
   python_warpx
   python_portable
