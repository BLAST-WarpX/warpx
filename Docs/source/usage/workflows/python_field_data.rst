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
The function ``sim.field.get`` returns a `pyamrex <https://pyamrex.readthedocs.io/en/latest/index.html>`__ object of type `MultiFab <https://pyamrex.readthedocs.io/en/latest/usage/api.html#amrex.space3d.MultiFab>`__, whose field data can be accessed or modified as described below.

Accessing/modifying the underlying field data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Several ways to access and modify the field data (i.e., the values of the fields on the grid points) are available.
These different methods differ in their user-friendliness, flexibility and performance overhead.

.. tab-set::

    .. tab-item:: Pre-defined AMReX methods

        The AMReX library defines many functions that can operate on field data, and many of them are accessible
        from Python via the `pyamrex <https://pyamrex.readthedocs.io/en/latest/index.html>`__ library. For a
        full list of these methods, see the `pyamrex API documentation <https://pyamrex.readthedocs.io/en/latest/usage/api.html#amrex.space3d.MultiFab>`__.

        Examples include:

        - Finding the maximum value of the field over the entire domain: ``Ex.max()``
        - Scaling the field by a factor of 2: ``Ex.mult(2.)``
        - Adding two fields together: ``Ex.saxpy(...)`` (see `this link <https://pyamrex.readthedocs.io/en/latest/usage/api.html#amrex.space3d.MultiFab.saxpy>`__)

        .. note::

            These methods generally have high performance and low overhead, including when using GPUs and multi-node parallelization, but are limited to the existing functions provided by AMReX.

        .. dropdown:: See some of these methods used in a full example

            .. literalinclude:: ../../../../Examples/Physics_applications/spacecraft_charging/inputs_test_rz_spacecraft_charging_picmi.py
                :language: python3
                :caption: You can copy this file from ``Examples/Physics_applications/spacecraft_charging/inputs_test_rz_secondary_ion_emission_picmi.py``.

    .. tab-item:: Numpy-like global indexing

        The field data in a ``MultiFab`` object can also be accessed via global indexing.
        Using standard array indexing with square brackets, the data can be accessed using indices that are relative to the full domain (across the ``MultiFab`` and across processors).
        When the data is fetched the result is a ``numpy`` array that contains a copy of the data, and when using multiple processors is broadcast to all processors (and is a global operation).
        For indices within the domain, values from valid cells are always returned.
        The ghost cells at the exterior of the domain are accessed using imaginary numbers, with negative values accessing the lower ghost cells, and positive the upper ghost cells.
        This example will return the ``Bz`` field at all valid interior points along ``x`` at the specified ``y`` and ``z`` indices.

        .. code-block:: python

            Bz = sim.fields.get("Bfield_fp", dir=2, level=0)
            Bz_along_x = Bz[:,5,6]

        The same global indexing can be done to set values. This example will set the values over a range in ``y`` and ``z`` at the
        specified ``x``. The data will be scattered appropriately to the underlying FABs. Setting values is a local operation.

        .. code-block:: python

            Jy = sim.fields.get("current_fp", dir=1, level=0)
            Jy[5,6:20,8:30] = 7.

        In the example below, 7 is added to all of the values along ``x``, including both valid and ghost cells (specified by using the empty tuple,
        ``()``), the first ghost cell at the lower boundary in ``y``, and the last valid cell and first upper ghost cell in ``z``.
        Note that the ``+=`` will be a global operation.

        .. code-block:: python

            Jx = sim.fields.get("current_fp", dir=0, level=0)
            Jx[(),-1j,-1:2j] += 7.

        Instead of setting values with a scalar value, you can also set values using an array. The array shape must match the selected region.
        The array must be either a ``numpy`` array (if WarpX is run on CPU) or a ``cupy`` array (if WarpX is run on GPU).
        To write portable code that works on both CPU and GPU, it is recommended to use the `load_cupy` package.
        See :ref:`usage-python-portable` for more details on writing portable Python code.

        .. code-block:: python

            from pywarpx.LoadThirdParty import load_cupy
            xp = load_cupy()

            Jy = sim.fields.get("current_fp", dir=1, level=0)
            # Create random values with shape matching the selected region
            random_values = xp.random.random((14, 22))  # shape: (20-6, 30-8)
            Jy[5,6:20,8:30] = random_values

        To fetch the data from all of the valid cells of all dimensions, the ellipsis can be used, ``Jx[...]``.
        Similarly, to fetch all of the data including valid cells and ghost cells, use an empty tuple, ``Jx[()]``.
        The code does error checking to ensure that the specified indices are within the bounds of the global domain.

        Finally, the ``mesh`` method returns the physical coordinates of the mesh along a specified direction,
        with appropriate centering based on the field's staggering. This is useful for plotting,
        analysis, or when you need to know the physical positions corresponding to field values.

        .. code-block:: python

            Ex = sim.field.get("Efield_fp", dir="x", level=0)
            x_coords = Ex.mesh("x")
            y_coords = Ex.mesh("y")
            z_coords = Ex.mesh("z")

        The method accepts a direction string (``"x"``, ``"y"``, ``"z"`` in 3D; ``"r"``, ``"z"`` in RZ geometry)
        and an optional ``include_ghosts`` parameter (default ``False``) to include ghost cell coordinates.
        The returned array contains the physical coordinates of the mesh points along the specified direction,
        properly accounting for the field's cell-centered or face-centered staggering.

        .. note::

            The global indexing is not the most high-performance way to access the field data
            (since it potentially incurs MPI communications and CPU-GPU copies under the hood),
            but it is the most flexible and easy to use.

    .. tab-item:: Explicit loop over boxes

        This method provides similar capabilities to the numpy-like global indexing approach, but operates
        only on local data within each MPI rank. Unlike global indexing, which may involve MPI communications
        and CPU-GPU data transfers under the hood, this approach performs all operations locally on each processor.
        As a result, this method offers significantly higher performance, especially for large-scale parallel simulations
        and GPU-accelerated runs. The data is accessed by explicitly looping over mesh-refinement levels and
        individual grid blocks (boxes), giving you direct access to the underlying ``numpy`` or ``cupy`` arrays for each local block.

        The example below accesses the :math:`Ex(x,y,z)` field at level 0 after every time step and sets all of the values to ``42``.
        This shows how to loop over levels and grid blocks.

        .. code-block:: python3

            from pywarpx import picmi
            from pywarpx.callbacks import callfromafterstep

            # Preparation: set up the simulation
            #   sim = picmi.Simulation(...)
            #   ...


            @callfromafterstep
            def set_Ex():
                warpx = sim.extension.warpx

                # data access
                #   vector field E, component x, on the fine patch of MR level 0
                Ex = sim.fields.get("Efield_fp", dir=0, level=0)
                #   scalar field rho, on the fine patch of MR level 0
                rho_mf = sim.fields.get("rho_fp", level=0)

                # compute on Ex
                # iterate over mesh-refinement levels
                for lev in range(warpx.finest_level + 1):
                    # grow (aka guard/ghost/halo) regions
                    ngv = Ex.n_grow_vect

                    # get every local block of the field
                    for mfi in Ex:
                        # global index space box, including guards
                        bx = mfi.tilebox().grow(ngv)
                        print(bx)  # note: global index space of this block

                        # numpy/cupy representation: non-copying view, including
                        # the guard/ghost region
                        Ex_arr = Ex.array(mfi).to_xp()

                        # notes on indexing in Ex:
                        # - numpy/cupy use locally zero-based indexing
                        # - layout is F_CONTIGUOUS by default, just like AMReX

                        # notes:
                        # Only the next lines are the "HOT LOOP" of the computation.
                        # For efficiency, we use array operation for speed.
                        Ex_arr[()] = 42.0


            sim.step(nsteps=100)

        For further details on how to `access GPU data <https://pyamrex.readthedocs.io/en/latest/usage/zerocopy.html>`__ or compute on ``Ex``, please see the `pyAMReX documentation <https://pyamrex.readthedocs.io/en/latest/usage/compute.html#fields>`__.


Defining a new custom field
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For some use cases, it is sometimes needed to create new custom fields (in addition to the :ref:`existing fields in WarpX <developers-fields-names>`).
New ``MultiFab`` objects can be created at the Python level. Using this method, the new ``MultiFab`` will be handled in the same way as WarpX's internal ``MultiFab``.
For example, their data will be automatically redistributed during load balancing (when the flags are set as shown in the example).

In the example below, a new ``MultiFab`` is created with the same properties as ``Ex``.

.. code-block:: python

   Ex = sim.fields.get("Efield_fp", dir=0, level=0)
   normalized_Ex = sim.fields.alloc_init(name="normalized_Ex",
                                         dir=0,
                                         level=0,
                                         ba=Ex.box_array(),
                                         dm=Ex.dm(),
                                         ncomp=Ex.n_comp,
                                         ngrow=Ex.n_grow_vect,
                                         initial_value=0.,
                                         redistribute=True,
                                         redistribute_on_remake=True)


.. dropdown:: See this function used in a full example

    .. literalinclude:: ../../../../Examples/Physics_applications/spacecraft_charging/inputs_test_rz_spacecraft_charging_picmi.py
        :language: python3
        :caption: You can copy this file from ``Examples/Physics_applications/spacecraft_charging/inputs_test_rz_secondary_ion_emission_picmi.py``.
