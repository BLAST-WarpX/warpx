Accessing the particles that hit the boundaries
-----------------------------------------------

WarpX can automatically save the particles that hit the boundaries
(see ``save_particles_at_xlo/ylo/zlo``, ``save_particles_at_xhi/yhi/zhi``,
and ``save_particles_at_eb`` in :ref:`running-cpp-parameters`).
This data can be accessed in Python via the ``ParticleBoundaryBufferWrapper`` object,
which can is initialized as shown below.

.. code-block:: python

    from pywarpx import particle_containers
    buffer = particle_containers.ParticleBoundaryBufferWrapper()

The ``ParticleBoundaryBufferWrapper`` object provides the following methods to access the particle boundary buffer data:

.. autoclass:: pywarpx.particle_containers.ParticleBoundaryBufferWrapper
   :members:

This can be used to implement physics at occurs at the boundary (e.g., secondary emission),
as in the example below.

.. dropdown:: See this function used in a full example

    .. literalinclude:: ../../../../Examples/Tests/secondary_ion_emission/inputs_test_rz_secondary_ion_emission_picmi.py
        :language: python3
        :caption: You can copy this file from ``Examples/Tests/secondary_ion_emission/inputs_test_rz_secondary_ion_emission_picmi.py``.
