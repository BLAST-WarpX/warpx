Accessing fields data
---------------------

Selecting a given field
^^^^^^^^^^^^^^^^^^^^^^^

The simulation's fields are accessed through the ``sim.field`` attribute, where ``sim`` is obtained as shown
in the :ref:`usage-python-extend-run-simulation` section. Specific fields (e.g. electric field, charge density, etc.)
are selected with the ``sim.field.get`` method, as shown in the example below.

.. code-block:: python

    # Preparation: set up the sim object
    #   sim = picmi.Simulation(...)
    #   ...

    # Extract the Ex field, at level 0 of mesh refinement
    Ex = sim.field.get("Efield_fp", dir="x", level=0)

The available field names (e.g. ``"Efield_fp"``, ``"rho_fp"``, etc.) are listed in the :ref:`developers-fields-names` section.
The function ``sim.field.get`` returns a `pyamrex <https://pyamrex.readthedocs.io/en/latest/index.html>`__ object of type ``MultiFab``, whose field data can be accessed or modified as described further below.

Accessing/modifying the underlying field data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Defining a new custom field
^^^^^^^^^^^^^^^^^^^^^^^^^^^
