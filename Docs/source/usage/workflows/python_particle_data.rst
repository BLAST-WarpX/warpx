Accessing particles data
------------------------

Selecting a given particle species
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simulation's particles are accessed through the ``sim.particles`` attribute, where ``sim`` is obtained as shown
in the :ref:`usage-python-extend-run-simulation` section. Specific particle species (e.g. electrons, ions, photons, etc.)
are selected with the ``sim.particles.get`` method, as shown in the example below.

.. code-block:: python


    # Preparation: set up the sim object
    #   sim = picmi.Simulation(...)
    #   ...

    # Extract the electrons particle species
    electrons = sim.particles.get("electrons")

The function ``sim.particles.get`` returns an object of type ``WarpXParticleContainer``, from which the
data of individual particles can be accessed or modified as described below.

Accessing/modifying the underlying particle data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Several ways to access and modify the particle data exist.
These different methods differ in their user-friendliness, flexibility and performance overhead.
(For more information, see the `pyamrex documentation <http://pyamrex.readthedocs.io/en/latest/usage/compute.html#particles>`__)

.. tab-set::

    .. tab-item:: Pandas-like global access (read-only)

        Gets all particles, irrespective of MPI ranks and mesh refinement levels
        Convenient to get all particles rapidly

        .. warning::
            Emphasize read-only

        .. note::
            Performance no good, because of MPI communications, CPU-GPU copies

    .. tab-item:: Explicit loop over boxes

        Local to mesh refinement level and box
        High performance

        Discuss cupy/numpy

Adding new particles
^^^^^^^^^^^^^^^^^^^^
