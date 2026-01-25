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

Gather some examples?

Document interface from
https://github.com/AMReX-Codes/pyamrex/pull/497

Functions:
- add_particles
- get_particle_count
- remove_particles




Particles
^^^^^^^^^

.. code-block:: python3

   from pywarpx import picmi
   from pywarpx.callbacks import callfromafterstep

   # Preparation: set up the simulation
   #   sim = picmi.Simulation(...)
   #   ...

   @callfromafterstep
   def my_after_step_callback():
       warpx = sim.extension.warpx
       Config = sim.extension.Config

       # data access
       multi_pc = warpx.multi_particle_container()
       pc = multi_pc.get_particle_container_from_name("electrons")

       # compute
       # iterate over mesh-refinement levels
       for lvl in range(pc.finest_level + 1):
           # get every local chunk of particles
           for pti in pc.iterator(pc, level=lvl):
               # compile-time and runtime attributes in SoA format
               soa = pti.soa().to_cupy() if Config.have_gpu else \
                     pti.soa().to_numpy()

               # notes:
               # Only the next lines are the "HOT LOOP" of the computation.
               # For speed, use array operation.

               # write to all particles in the chunk
               # note: careful, if you change particle positions, you might need to
               #       redistribute particles before continuing the simulation step
               soa.real[0][()] = 0.30  # x
               soa.real[1][()] = 0.35  # y
               soa.real[2][()] = 0.40  # z

               # all other attributes: weight, momentum x, y, z, ...
               for soa_real in soa.real[3:]:
                   soa_real[()] = 42.0

               # by default empty unless ionization or QED physics is used
               # or other runtime attributes were added manually
               for soa_int in soa.int:
                   soa_int[()] = 12


   sim.step(nsteps=100)

For further details on how to `access GPU data <https://pyamrex.readthedocs.io/en/latest/usage/zerocopy.html>`__ or compute on ``electrons``, please see the `pyAMReX documentation <https://pyamrex.readthedocs.io/en/latest/usage/compute.html#particles>`__.


High-Level Particle Wrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   TODO: What are the benefits of using the high-level wrapper?
   TODO: What are the limitations (e.g., in memory usage or compute scalability) of using the high-level wrapper?

Particles can be added to the simulation at specific positions and with specific attribute values:

.. code-block:: python

   from pywarpx import particle_containers, picmi

   # ...

   electron_wrapper = particle_containers.ParticleContainerWrapper("electrons")


.. autoclass:: pywarpx.particle_containers.ParticleContainerWrapper
   :members:

The ``get_particle_real_arrays()``, ``get_particle_int_arrays()`` and
``get_particle_idcpu_arrays()`` functions are called
by several utility functions of the form ``get_particle_{comp_name}`` where
``comp_name`` is one of ``x``, ``y``, ``z``, ``r``, ``theta``, ``id``, ``cpu``,
``weight``, ``ux``, ``uy`` or ``uz``.
