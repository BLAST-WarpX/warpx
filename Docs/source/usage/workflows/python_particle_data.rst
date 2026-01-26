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

Quantities available
 one of x, y, z, r, theta, id, cpu, weight, ux, uy or uz.

.. tab-set::

    .. tab-item:: Pandas-like global access (read-only)

        Gets all particles, irrespective of MPI ranks and mesh refinement levels
        Convenient to get all particles rapidly

        .. code-block:: python

            # code-specific getter function, e.g.:
            # pc = sim.get_particles()
            # Config = sim.extension.Config

            # local particles on all levels
            df = pc.to_df()  # this is a copy!
            print(df)

            # read
            print(df["x"])

            # write (into copy!)
            df["x"] = 0.30
            df["y"] = 0.35
            df["z"] = 0.40

            df["a"] = df["x"] ** 2
            df["b"] = df["x"] + df["y"]
            df["c"] = 0.50

            # int attributes
            # df["i1"] = 12
            # df["i2"] = 12
            # ...

            print(df)

        .. warning::
            Emphasize read-only

        .. note::
            Performance no good, because of MPI communications, CPU-GPU copies

    .. tab-item:: Explicit loop over boxes

        Local to mesh refinement level and box
        High performance

        .. code-block:: python

            # code-specific getter function, e.g.:
            # pc = sim.get_particles()
            # Config = sim.extension.Config

            # iterate over particles on level 0
            for pti in pc.iterator(level=0):
                # print all particle ids in the tile
                print("idcpu =", pti["idcpu"])

                x = pti["x"]  # this is automatically a cupy or numpy
                y = pti["y"]  #   array, depending on Config.have_gpu

                # write to all particles in the chunk
                # note: careful, if you change particle positions, you might need to
                #       redistribute particles before continuing the simulation step
                pti["x"][:] = 0.30
                pti["y"][:] = 0.35
                pti["z"][:] = 0.40

                pti["a"][:] = x[:] ** 2
                pti["b"][:] = x[:] + y[:]
                pti["c"][:] = 0.50
                # ...

                # int attributes
                pti["i1"][:] = 12
                pti["i2"][:] = 13

        Discuss cupy/numpy

Adding new particles
^^^^^^^^^^^^^^^^^^^^

New particles can be added by using the method ``add_particles`` of the ``WarpXParticleContainer`` object.
This method takes the following arguments:


The method returns the number of added particles.

.. dropdown:: See this function used in a full example

    .. literalinclude:: ../../../../Examples/Tests/particle_boundary_interaction/inputs_test_rz_particle_boundary_interaction_picmi.py
        :language: python
        :caption: You can copy this file from ``Examples/Tests/particle_boundary_interaction/inputs_test_rz_particle_boundary_interaction_picmi.py``.
