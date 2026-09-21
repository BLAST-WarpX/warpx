.. _developers-how-to-add-picmi-parameter:

How to Add a PICMI Parameter
============================

This guide is for developers who added an input parameter in C++ and now need to expose it to users of the :ref:`Python (PICMI) interface <usage-picmi>`.

In C++, WarpX reads its :ref:`input parameters <running-cpp-parameters>` with ``amrex::ParmParse``, for example ``warpx.my_threshold`` with:

.. code-block:: cpp

   const amrex::ParmParse pp_warpx("warpx");
   utils::parser::queryWithParser(pp_warpx, "my_threshold", m_my_threshold);

In Python, users set up a simulation with PICMI classes, e.g., ``picmi.Simulation``, which write these input parameters for WarpX (see :ref:`development-python`).
Parameters of the PICMI standard have plain names, e.g., ``max_steps``.
Parameters that only WarpX has are *extensions*, with the prefix ``warpx_``, e.g.:

.. code-block:: python

   sim = picmi.Simulation(max_steps=100, warpx_my_threshold=0.2)

The PICMI classes in ``Python/pywarpx/picmi.py`` are `pydantic <https://docs.pydantic.dev>`__ models:
each parameter is a *field* with a type and a description.
Pydantic checks the values that users give, and the :ref:`documentation of the PICMI classes <usage-picmi-parameters>` is generated from the fields.

.. note::

   Users who pass a parameter that a PICMI class does not have get an error, e.g., ``warpx_my_threshold: Extra inputs are not permitted``.
   So a new C++ parameter is only available in Python once it is added to the PICMI class as described below.


1. Find the PICMI class
-----------------------

The prefix of the input parameter determines the PICMI class, and the method of that class that writes the inputs of this prefix:

.. list-table::
   :header-rows: 1

   * - Input parameters
     - PICMI class
     - Method that writes them
   * - ``warpx.*``, ``algo.*``, ``amr.*``, ``particles.*`` (general)
     - ``Simulation``
     - ``initialize_inputs``
   * - ``geometry.*``, ``boundary.*``, ``amr.*`` (grid)
     - ``Cartesian1DGrid``, ``Cartesian2DGrid``, ``Cartesian3DGrid``, ``CylindricalGrid``
     - ``grid_initialize_inputs``
   * - ``algo.*``, ``psatd.*``, ``hybrid_pic_model.*``, ``warpx.*`` (field solver)
     - ``ElectromagneticSolver``, ``ElectrostaticSolver``, ``HybridPICSolver``
     - ``solver_initialize_inputs``
   * - ``<species name>.*``
     - ``Species``, the distributions (e.g., ``UniformDistribution``)
     - ``species_initialize_inputs``, ``distribution_initialize_inputs``
   * - ``<diagnostic name>.*``
     - ``FieldDiagnostic``, ``ParticleDiagnostic``, ``ReducedDiagnostic``, ...
     - ``diagnostic_initialize_inputs``
   * - ``<laser name>.*``
     - ``GaussianLaser``, ``AnalyticLaser``, ``LaserAntenna``
     - ``laser_initialize_inputs``, ``laser_antenna_initialize_inputs``
   * - ``<collision name>.*``
     - ``CoulombCollisions``, ``MCCCollisions``, ...
     - ``collision_initialize_inputs``

When in doubt, look for existing parameters of the same prefix in ``Python/pywarpx/picmi.py``, e.g., for the prefix ``warpx``:

.. code-block:: bash

   grep -n "pywarpx.warpx\." Python/pywarpx/picmi.py

Choose the class whose objects the parameter belongs to, e.g., a parameter of the PSATD solver belongs to ``ElectromagneticSolver``, even though it could also be written by ``Simulation``.


2. Add a field
--------------

Add the parameter as a field of the class, next to related fields, with the name that it has in the inputs (without the prefix):

.. code-block:: python

   class Simulation(picmistandard.PICMI_Simulation):
       ...
       my_threshold: float | None = Field(
           default=None,
           ge=0.0,
           description="Fraction of ... above which ... [unit]",
       )

Give the field:

* **A type** that matches the C++ parameter (see the table below), with ``| None``.
* **The default** ``None``, which means that the user did not set the parameter.
  WarpX then does not write it to the inputs, so that the default in C++ applies.
  Only give a different default if Python needs to override the C++ default, and explain why.
* **Constraints** if the values are limited, e.g., ``ge=0.0`` (``>= 0``), ``gt``, ``le``, ``lt``, or a ``Literal`` of the allowed choices.
  Pydantic then rejects invalid values when users create the object or change the parameter later on.
* **A description** of the parameter, with its unit, as shown in the :ref:`documentation <usage-picmi-parameters>`.
  Use the same wording as in the :ref:`C++ parameter documentation <running-cpp-parameters>`.

.. list-table::
   :header-rows: 1

   * - C++ parameter
     - Python type
     - Written to the inputs as
   * - ``bool``
     - ``bool | None``
     - ``1`` or ``0``
   * - ``int``
     - ``int | None``
     - ``3``
   * - ``amrex::Real``
     - ``float | None``
     - ``0.001``
   * - number read with ``queryWithParser``, which can also be an expression of ``my_constants``
     - ``float | str | None``
     - ``0.001`` or ``"2*dx"``
   * - string with a fixed set of choices
     - ``Literal["first", "second"] | None``
     - ``"first"``
   * - array (``queryarr``)
     - ``list[int] | None``, ``list[float] | None``, ``list[str] | None``
     - ``1 1 2``
   * - parser function, e.g., ``f(x,y,z,t)``
     - ``Expression | None``
     - ``"x*2"``

For a parser function, follow an existing expression parameter of the same class:
constants that users give with the expression are collected, and the method that writes the inputs adds them to ``my_constants``.

The ``warpx_`` prefix
^^^^^^^^^^^^^^^^^^^^^

Users give an extension of a PICMI standard class with the prefix ``warpx_``, e.g., ``warpx_my_threshold``.
The field itself has the name without prefix, which is also the name that the methods of the class use, e.g., ``self.my_threshold``.
The prefix is added to all fields that a WarpX class adds to the PICMI standard class by

.. code-block:: python

   model_config = ConfigDict(alias_generator=warpx_options(picmistandard.PICMI_Simulation))

in the class.
If the class does not have this line yet, add it (with the PICMI standard class that it derives from).

There are two exceptions:

* Classes that only exist in WarpX, which derive from an extension class of the standard (``picmistandard.PICMI_Extension``, ``PICMI_SolverExtension``, ``PICMI_DiagnosticExtension``, ...), e.g., ``HybridPICSolver`` or ``ReducedDiagnostic``, use no prefix: all their parameters are WarpX parameters.
* If the name that users give is not ``warpx_`` followed by the name of the field, give it explicitly, e.g., ``Field(default=None, alias="warpx_potential_lo_x", ...)``.


3. Write the input parameter
----------------------------

In the method of the class that writes its inputs (see step 1), assign the field to the prefix:

.. code-block:: python

   def initialize_inputs(self):
       ...
       pywarpx.warpx.my_threshold = self.my_threshold

This writes the line ``warpx.my_threshold = 0.2`` to the inputs, and nothing if the value is ``None``.
The prefixes of objects with a name, e.g., species and diagnostics, are attributes of the objects, e.g., ``self._species.my_flag = self.my_flag`` in ``Species``.


4. Test
-------

Check the inputs that the Python class writes, without building WarpX:
write a small script that sets the parameter, e.g., ``check.py``,

.. code-block:: python

   from pywarpx import picmi

   grid = picmi.Cartesian3DGrid(
       number_of_cells=[16, 16, 16],
       lower_bound=[0.0, 0.0, 0.0],
       upper_bound=[1.0, 1.0, 1.0],
       lower_boundary_conditions=["periodic"] * 3,
       upper_boundary_conditions=["periodic"] * 3,
   )
   solver = picmi.ElectromagneticSolver(grid=grid, cfl=0.99)
   sim = picmi.Simulation(solver=solver, max_steps=1, warpx_my_threshold=0.2)
   sim.write_input_file("inputs_check")

and run it with the Python files of your source tree:

.. code-block:: bash

   PYTHONPATH=Python python3 check.py
   grep my_threshold inputs_check

Also check that an invalid value, e.g., ``warpx_my_threshold=-1.0``, raises a ``ValidationError``.

Then use the parameter in a test of the parameter's feature, in the PICMI input script of an existing test (``Examples/**/inputs_test_*_picmi.py``) or in a new one, and run it as described in :ref:`developers-testing`.
This needs a build of WarpX with ``-DWarpX_PYTHON=ON``.


5. Check the documentation
--------------------------

The :ref:`documentation of the PICMI classes <usage-picmi-parameters>` shows the new field with its type, default and description, e.g., as ``warpx_my_threshold``.
Nothing needs to be added to ``Docs/source/usage/python.rst``.
To check the documentation, build it as described in :ref:`developers-docs`.

The C++ parameter itself is documented in ``Docs/source/usage/parameters.rst``, as part of the C++ change.


Checklist
---------

* [ ] Field added to the PICMI class of the parameter, with type, default ``None``, constraints and description.
* [ ] The class has the ``warpx_`` prefix for its WarpX fields (``warpx_options``), unless it is a WarpX-only class.
* [ ] The field is written in the method of the class that writes its inputs.
* [ ] The inputs contain the parameter (``write_input_file``), and invalid values are rejected.
* [ ] A PICMI test uses the parameter.
* [ ] ``pre-commit`` passes (see :ref:`developers-testing`).

.. tip::

   LLM coding assistants in the WarpX repository can follow the skill ``/warpx-add-picmi-parameter``, which implements the steps above (see :ref:`developers-llm`).
