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

There are several ways to access and modify the particle data, i.e. the positions (``'x'``, ``'y'``, ``'z'`` in 3D Cartesian geometry),
normalized momenta (``'ux'``, ``'uy'``, ``'uz'``), the particle weights (``'w'``), and the unique IDs (``'idcpu'``) of each individual particles.

.. note::

   In geometries other than 3D Cartesian, particle positions are defined by different variables.
   For example, in RZ geometry, positions are accessed as ``'r'`` and ``'z'`` (with an additional ``'theta'`` attribute),
   while in RCYLINDER geometry, only ``'r'`` is available (with ``'theta'``), and in RSPHERE geometry, ``'r'`` is available
   (with ``'theta'`` and ``'phi'``). See :ref:`developers-dimensionality` for a complete table of position attributes
   available in each geometry.

The different methods below differ in their user-friendliness, flexibility and performance overhead.
(For more information, see the `pyamrex documentation <http://pyamrex.readthedocs.io/en/latest/usage/compute.html#particles>`__.)

.. tab-set::

    .. tab-item:: Global access through pandas DataFrame (read-only)

        The method ``to_df`` of the ``WarpXParticleContainer`` object returns a
        `pandas DataFrame <https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe>`__ containing the particle data.
        More specifically, the keys of the DataFrame are the particle attributes (e.g., ``'ux'``, ``'w'``, ``'idcpu'``),
        and the corresponding arrays have one element per particle, and contain the particles of that species across all
        boxes and tiles (on the current MPI rank) and across all mesh refinement levels.

        .. warning::

            The data in the DataFrame is a copy of the particle data, and therefore modifying it will not modify
            the actual particle data in the simulation.

        .. note::

            The method ``to_df`` is very convenient because it automatically concatenates all particles across boxes and tiles,
            and across all mesh refinement levels. However, this implies significant performance overheads, as it incurs copies
            and CPU-GPU data transfers. This method is thus mostly meant for debugging and visualization purposes,
            and not for performance-critical operations.

        .. code-block:: python

            # Preparation: set up the simulation
            #   sim = picmi.Simulation(...)
            #   ...

            # Extract the electrons particle species
            electrons = sim.particles.get("electrons")

            # local particles (returns only particles on the current MPI rank)
            df = electrons.to_df(local=True)  # this is a copy!
            print('Available attributes: ', df.columns)
            print('Number of particles: ', len(df))

            # print position x (one element per particle)
            print('Position x: ', df['x'])

            # Warning: because `df` is a copy, modifying it will
            # not modify the actual particle data
            df['x'] += 0.1 # This does not modify the actual particle data

    .. tab-item:: Explicit loop over boxes/tiles

        This method provides similar capabilities to the pandas DataFrame approach, but adapts to
        the data structure of the particles in WarpX (i.e. particles are organized per box/tile and
        mesh refinement level). Unlike the pandas DataFrame approach, this avoids unneeded copies
        and CPU-GPU data transfers. As a result, this method offers significantly higher performance,
        especially for large-scale parallel simulations and GPU-accelerated runs. The data is accessed
        by explicitly looping over mesh-refinement levels and individual grid blocks (boxes), giving
        direct access to the underlying particle data arrays for each local block.

        .. code-block:: python

            # Preparation: set up the simulation
            #   sim = picmi.Simulation(...)
            #   xp, _ = load_cupy()
            #   ...

            # Extract the electrons particle species
            electrons = sim.particles.get("electrons")

            # iterate over boxes/tiles on level 0
            for pti in electrons.iterator(level=0):

                # print position x (one element per particle)
                print('Position x: ', pti['x'])

                # increment position by a random value
                # using numpy/cupy syntax
                pti["x"][:] += xp.random.random( len(pti['x'][:]) )

        In the above example, ``xp`` represents either the ``numpy`` or ``cupy`` package.
        See :ref:`usage-python-portable` for more details on writing portable Python code.

Adding new particles
^^^^^^^^^^^^^^^^^^^^

New particles can be added to a given species by using the method ``add_particles`` of the ``WarpXParticleContainer`` object,
using the following syntax:

.. autofunction:: pywarpx.extensions.WarpXParticleContainer.add_particles

.. dropdown:: See this function used in a full example

    .. literalinclude:: ../../../../Examples/Tests/particle_boundary_interaction/inputs_test_rz_particle_boundary_interaction_picmi.py
        :language: python
        :caption: You can copy this file from ``Examples/Tests/particle_boundary_interaction/inputs_test_rz_particle_boundary_interaction_picmi.py``.
