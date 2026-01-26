.. _usage-python-portable:

Writing portable Python code that can be executed on CPU and GPU
----------------------------------------------------------------

When accessing field/particle data in Python, access is exposed through array-like structures.
Depending on whether WarpX is running with GPU support or not, these arrays are stored either on CPU or GPU.
Working with those arrays requires a Python package that operates on CPU (e.g. `numpy <https://numpy.org/doc/stable/>`__) or GPU (e.g. `cupy <https://docs.cupy.dev/en/stable/>`__).
Note that ``numpy`` and ``cupy`` have almost identical syntax, making it easy to write portable code that is not specific to CPU or GPU.
In order to do so, one needs a functionality that will automatically detect whether WarpX runs on CPU or GPU and import the package ``numpy`` or ``cupy`` accordingly.
This functionality is provided by the function :func:`load_cupy`, which can be used as shown below.

.. code-block:: python

      from pywarpx.LoadThirdParty import load_cupy
      xp, _ = load_cupy()

In this example, the ``xp`` variable is either ``numpy`` (often abbreviated as ``np``) or ``cupy`` (often abbreviated as ``cp``), depending on whether WarpX is running with GPU support or not.

.. dropdown:: See this used in a full example

   .. literalinclude:: ../../../../Examples/Tests/particle_boundary_interaction/inputs_test_rz_particle_boundary_interaction_picmi.py
         :language: python3
         :caption: You can copy this file from ``Examples/Physics_applications/spacecraft_charging/inputs_test_rz_secondary_ion_emission_picmi.py``.
