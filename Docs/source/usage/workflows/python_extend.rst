.. _usage-python-extend:

Extend a Simulation with Python
===============================

When running WarpX directly :ref:`from Python <usage-picmi>` it is possible to interact with the simulation.

For instance, with the :py:meth:`~pywarpx.picmi.Simulation.step` method of the simulation class, one could run ``sim.step(nsteps=1)`` in a loop:

.. code-block:: python3

   # Preparation: set up the simulation
   #   sim = picmi.Simulation(...)
   #   ...

   steps = 1000
   for _ in range(steps):
       sim.step(nsteps=1)

       # do something custom with the sim object

As a more flexible alternative, one can install `callback functions <https://en.wikipedia.org/wiki/Callback_(computer_programming)>`__, which will execute a given Python function at a
specific location in the WarpX simulation loop.

.. automodule:: pywarpx.callbacks
   :members: installcallback, uninstallcallback, isinstalled


pyAMReX
-------

Many of the following classes are provided through `pyAMReX <https://github.com/AMReX-Codes/pyamrex>`__.
After the simulation is initialized, the pyAMReX module can be accessed via

.. code-block:: python

   from pywarpx import picmi, libwarpx

   # ... simulation definition ...

   # equivalent to
   #   import amrex.space3d as amr
   # for a 3D simulation
   amr = libwarpx.amr  # picks the right 1d, 2d or 3d variant


Full details for pyAMReX APIs are `documented here <https://pyamrex.readthedocs.io/en/latest/usage/api.html>`__.
Important APIs include:

* `amr.ParallelDescriptor <https://pyamrex.readthedocs.io/en/latest/usage/api.html#amrex.space3d.ParallelDescriptor.IOProcessor>`__: MPI-parallel rank information
* `amr.MultiFab <https://pyamrex.readthedocs.io/en/latest/usage/api.html#amrex.space3d.MultiFab>`__: MPI-parallel field data
* `amr.ParticleContainer_* <https://pyamrex.readthedocs.io/en/latest/usage/api.html#amrex.space3d.ParticleContainer_1_1_2_1_default>`__: MPI-parallel particle data for a particle species


Data Access
-----------

While the simulation is running, callbacks can have read and write access the WarpX simulation data *in situ*.

An important object in the ``pywarpx.picmi`` module for data access is ``Simulation.extension.warpx``, which is available only during the simulation run.
This object is the Python equivalent to the C++ ``WarpX`` simulation class.

.. py:class:: WarpX

   .. py:method:: getistep(lev: int)

      Get the current step on mesh-refinement level ``lev``.

   .. py:method:: gett_new(lev: int)

      Get the current physical time on mesh-refinement level ``lev``.

   .. py:method:: getdt(lev: int)

      Get the current physical time step size on mesh-refinement level ``lev``.

   .. py:method:: multifab(multifab_name: str)

      Return MultiFabs by name, e.g., ``"Efield_aux[x][level=0]"``, ``"Efield_cp[x][level=0]"``, ...

      The physical fields in WarpX have the following naming:

      - ``_fp`` are the "fine" patches, the regular resolution of a current mesh-refinement level
      - ``_aux`` are temporary (auxiliar) patches at the same resolution as ``_fp``.
        They usually include contributions from other levels and can be interpolated for gather routines of particles.
      - ``_cp`` are "coarse" patches, at the same resolution (but not necessary values) as the ``_fp`` of ``level - 1``
        (only for level 1 and higher).

   .. py:method:: multi_particle_container

   .. py:method:: get_particle_boundary_buffer

   .. py:method:: set_potential_on_domain_boundary(potential_[lo/hi]_[x/y/z]: str)

      The potential on the domain boundaries can be modified when using the electrostatic solver.
      This function updates the strings and function parsers which set the domain
      boundary potentials during the Poisson solve.

   .. py:method:: set_potential_on_eb(potential: str)

      The embedded boundary (EB) conditions can be modified when using the electrostatic solver.
      This set the EB potential string and updates the function parser.

   .. py:method:: evolve(numsteps=-1)

      Evolve the simulation the specified number of steps.

   .. autofunction:: pywarpx.picmi.Simulation.extension.finalize

.. py::def:: pywarpx.picmi.Simulation.extension.get_instance

   Return a reference to the WarpX object.


The :py:class:`WarpX` also provides read and write access to field ``MultiFab`` and ``ParticleContainer`` data, shown in the following examples.

Fields
^^^^^^

This example accesses the :math:`E_x(x,y,z)` field at level 0 after every time step and calculate the largest value in it.

.. code-block:: python3

   from pywarpx import picmi
   from pywarpx.callbacks import callfromafterstep

   # Preparation: set up the simulation
   #   sim = picmi.Simulation(...)
   #   ...


   @callfromafterstep
   def set_E_x():
       warpx = sim.extension.warpx
       multifab_register = warpx.multifab_register()

       # data access
       #   vector field E, component x, on the fine patch of MR level 0
       E_x_mf = multifab_register.get("Efield_fp", dir=0, level=0)
       #   scalar field rho, on the fine patch of MR level 0
       rho_mf = multifab_register.get("rho_fp", level=0)

       # compute on E_x_mf
       # iterate over mesh-refinement levels
       for lev in range(warpx.finest_level + 1):
           # grow (aka guard/ghost/halo) regions
           ngv = E_x_mf.n_grow_vect

           # get every local block of the field
           for mfi in E_x_mf:
               # global index space box, including guards
               bx = mfi.tilebox().grow(ngv)
               print(bx)  # note: global index space of this block

               # numpy representation: non-copying view, including the
               # guard/ghost region;     .to_cupy() for GPU!
               E_x_np = E_x_mf.array(mfi).to_numpy()

               # notes on indexing in E_x_np:
               # - numpy uses locally zero-based indexing
               # - layout is F_CONTIGUOUS by default, just like AMReX

               # notes:
               # Only the next lines are the "HOT LOOP" of the computation.
               # For efficiency, use numpy array operation for speed on CPUs.
               # For GPUs use .to_cupy() above and compute with cupy or numba.
               E_x_np[()] = 42.0


   sim.step(nsteps=100)

For further details on how to `access GPU data <https://pyamrex.readthedocs.io/en/latest/usage/zerocopy.html>`__ or compute on ``E_x``, please see the `pyAMReX documentation <https://pyamrex.readthedocs.io/en/latest/usage/compute.html#fields>`__.

A warning is that it is recommended that the reference to the MultiFab returned by multifab_register.get should not be saved across time steps. If there is load balancing, the MultiFabs will be regenerated and that reference will become invalid.

High-Level Field Wrapper
^^^^^^^^^^^^^^^^^^^^^^^^

The ``fields`` module provides wrappers around the MultiFabs that are defined in the WarpX class, those that are added to the MultiFab registry.
For a list of all of the available wrappers, see the file ``Python/pywarpx/fields.py``.
For each MultiFab, there is a function that will return a wrapper around the data.
The wrappers provide a convenient interface to the MultiFabs, which have the advantage that they can be used when load balancing is done.
For example, the function ``ExWrapper`` returns a wrapper around the ``x`` component of the MultiFab vector ``Efield_aux`` at level 0.

.. code-block:: python

   from pywarpx import fields
   Ex = fields.ExWrapper(level=0)

The wrapper provides access to the data via global indexing.
Using standard array indexing with square brackets, the data can be accessed using indices that are relative to the full domain (across the MultiFab and across processors).
When the data is fetched the result is a numpy array that contains a copy of the data, and when using multiple processors is broadcast to all processors (and is a global operation).
For indices within the domain, values from valid cells are always returned.
The ghost cells at the exterior of the domain are accessed using imaginary numbers, with negative values accessing the lower ghost cells, and positive the upper ghost cells.
This example will return the ``Bz`` field at all valid interior points along ``x`` at the specified ``y`` and ``z`` indices.

.. code-block:: python

   from pywarpx import fields
   Bz = fields.BzWrapper()
   Bz_along_x = Bz[:,5,6]

The same global indexing can be done to set values. This example will set the values over a range in ``y`` and ``z`` at the
specified ``x``. The data will be scattered appropriately to the underlying FABs. The set is a local operation.

.. code-block:: python

   from pywarpx import fields
   Jy = fields.JyFPWrapper()
   Jy[5,6:20,8:30] = 7.

In this example, seven is added to all of the values along ``x``, including both valid and ghost cells (specified by using the empty tuple, ``()``), the first ghost cell at the lower boundary in ``y``, and the last valid cell and first upper ghost cell in ``z``.
Note that the ``+=`` will be a global operation.

.. code-block:: python

   from pywarpx import fields
   Jx = fields.JxFPWrapper()
   Jx[(),-1j,-1:2j] += 7.

To fetch the data from all of the valid cells of all dimensions, the ellipsis can be used, ``Jx[...]``.
Similarly, to fetch all of the data including valid cells and ghost cells, use an empty tuple, ``Jx[()]``.
The code does error checking to ensure that the specified indices are within the bounds of the global domain.

The wrapper allows new MultiFabs to be created at the Python level and added to the registry.
In this example, a new MultiFab is added with the same properties as `Ex`.

.. code-block:: python

   from pywarpx import fields
   Ex = fields.ExWrapper()
   normalized_Ex = fields.MultiFabWrapper(create_new=True,
                                          mf_name="normalized_Ex",
                                          idir=0,
                                          ba=Ex.box_array(),
                                          ngrow=Ex.n_grow_vect)

Under the covers, the wrapper object is using the Python wrapper of a MultiFab, relying on its global array indexing capability.
The wrappers are always up to date since whenever an access is done (either a get or a set), the wrapper fetches the underlying MultiFab object.

Through the wrapper, all of the operations available on the underlying MultiFab can be done. For example, to find the max value, use ``Jx.max()``, and to multiply the data by a factor, ``Jx.mult(2.)``.

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


Diagnostics
-----------

Various diagnostics are also accessible from Python.
This includes getting the deposited or total charge density from a given species as well as accessing the scraped particle buffer.
See the example in ``Examples/Tests/ParticleBoundaryScrape`` for a reference on how to interact with scraped particle data.


.. autoclass:: pywarpx.particle_containers.ParticleBoundaryBufferWrapper
   :members:


Modify Solvers
--------------

From Python, one can also replace numerical solvers in the PIC loop or add new physical processes into the time step loop.
Examples:

* :ref:`Capacitive Discharge <examples-capacitive-discharge>`: replaces the Poisson solver of an electrostatic simulation (default: MLMG) with a python function that uses `superLU <https://portal.nersc.gov/project/sparse/superlu/>`__ to directly solve the Poisson equation.
