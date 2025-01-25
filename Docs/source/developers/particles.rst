.. _developers-particles:

Particles
=========

Particle containers
-------------------

Particle structures and functions are defined in ``Source/Particles/``. WarpX uses the ``Particle`` class from AMReX for single particles. An ensemble of particles (e.g., a plasma species, or laser particles) is stored as a ``WarpXParticleContainer`` (see description below) in a per-box (and even per-tile on CPU) basis.

.. doxygenclass:: WarpXParticleContainer

Physical species are stored in ``PhysicalParticleContainer``, that derives from ``WarpXParticleContainer``. In particular, the main function to advance all particles in a physical species is ``PhysicalParticleContainer::Evolve`` (see below).

.. doxygenfunction:: PhysicalParticleContainer::Evolve
   :outline:

Finally, all particle species (physical plasma species ``PhysicalParticleContainer``, photon species ``PhotonParticleContainer`` or non-physical species ``LaserParticleContainer``) are stored in ``MultiParticleContainer``. The class ``WarpX`` holds one instance of ``MultiParticleContainer`` as a member variable, called ``WarpX::mypc`` (where `mypc` stands for "my particle containers"):

.. doxygenclass:: MultiParticleContainer

Loop over particles
-------------------

A typical loop over particles reads:

.. code-block:: cpp

  // pc is a std::unique_ptr<WarpXParticleContainer>
  // Loop over MR levels
  for (int lev = 0; lev <= finest_level; ++lev) {
      // Loop over particles, box by box
      for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti) {
          // Do something on particles
          // [MY INNER LOOP]
      }
  }

The innermost step ``[MY INNER LOOP]`` typically calls ``amrex::ParallelFor`` to perform operations on all particles in a portable way. The innermost loop in the code snippet above could look like:

.. code-block:: cpp

  // Get Struct-Of-Array particle data, also called attribs
  // (x, y, z, ux, uy, uz, w)
  auto& attribs = pti.GetAttribs();
  auto& x = attribs[PIdx::x];
  // [...]
  // Number of particles in this box
  const long np = pti.numParticles();

Link fields and particles?
--------------------------

In WarpX, the loop over boxes through a ``MultiFab`` iterator ``MFIter`` and the loop over boxes through a ``ParticleContainer`` iterator ``WarpXParIter`` are consistent.

On a loop over boxes in a ``MultiFab`` (``MFIter``), it can be useful to access particle data on a GPU-friendly way. This can be done by:

.. code-block:: cpp

  // Index of grid (= box)
  const int grid_id = mfi.index();
  // Index of tile within the grid
  const int tile_id = mfi.LocalTileIndex();
  // Get GPU-friendly arrays of particle data
  auto& ptile = GetParticles(lev)[std::make_pair(grid_id,tile_id)];
  // Only need attribs (i.e., SoA data)
  auto& soa = ptile.GetStructOfArrays();
  // As an example, let's get the ux momentum
  const ParticleReal * const AMREX_RESTRICT ux = soa.GetRealData(PIdx::ux).data();

On a loop over particles it can be useful to access the fields on the box we are looping over (typically when we use both field and particle data on the same box, for field gather or current deposition for instance). This is done for instance by adding this snippet in ``[MY INNER LOOP]``:

.. code-block:: cpp

  // E is a reference to, say, WarpX::Efield_aux
  // Get the Ex field on the grid
  const FArrayBox& exfab = (*E[lev][0])[pti];
  // Let's be generous and also get the underlying box (i.e., index info)
  const Box& box = pti.validbox();

Main functions
--------------

.. doxygenfunction:: PhysicalParticleContainer::PushPX

.. doxygenfunction:: WarpXParticleContainer::DepositCurrent(ablastr::fields::MultiLevelVectorField const &J, amrex::Real dt, amrex::Real relative_time)

.. note::
   The current deposition is used both by ``PhysicalParticleContainer`` and ``LaserParticleContainer``, so it is in the parent class ``WarpXParticleContainer``.

Buffers
-------

To reduce numerical artifacts at the boundary of a mesh-refinement patch, WarpX has an option to use buffers: When particles evolve on the fine level, they gather from the coarse level (e.g., ``Efield_cax``, a copy of the ``aux`` data from the level below) if they are located on the fine level but fewer than ``WarpX::n_field_gather_buffer`` cells away from the coarse-patch boundary. Similarly, when particles evolve on the fine level, they deposit on the coarse level (e.g., ``Efield_cp``) if they are located on the fine level but fewer than ``WarpX::n_current_deposition_buffer`` cells away from the coarse-patch boundary.

``WarpX::gather_buffer_masks`` and ``WarpX::current_buffer_masks`` contain masks indicating if a cell is in the interior of the fine-resolution patch or in the buffers. Then, particles depending on this mask in

.. doxygenfunction:: PhysicalParticleContainer::PartitionParticlesInBuffers

.. note::

   Buffers are complex!

Particle attributes
-------------------

WarpX adds the following particle attributes by default to WarpX particles.
These attributes are stored in Struct-of-Array (SoA) locations of the AMReX particle containers: one SoA for ``amrex::ParticleReal`` attributes, one SoA for ``int`` attributes and one SoA for a ``uint64_t`` global particle index per particle.
The data structures for those are either pre-described at compile-time (CT) or runtime (RT).

====================  ================  ==================================  ===== ==== ======================
Attribute name        ``int``/``real``  Description                         Where When Notes
====================  ================  ==================================  ===== ==== ======================
``position_x/y/z``    ``real``          Particle position.                  SoA   CT
``weight``            ``real``          Particle position.                  SoA   CT
``momentum_x/y/z``    ``real``          Particle position.                  SoA   CT
``id``                ``amrex::Long``   CPU-local particle index            SoA   CT   First 40 bytes of
                                        where the particle was created.                idcpu
``cpu``               ``int``           CPU index where the particle        SoA   CT   Last 24 bytes of idcpu
                                        was created.
``stepScraped``        ``int``          PIC iteration of the last step      SoA   RT   Added when there is
                                        before the particle hits the                   particle-boundary
                                        boundary.                                      interaction.
                                                                                       Saved in the boundary
                                                                                       buffers.
``deltaTimeScraped``   ``real``         Difference of time between the      SoA   RT   Added when there is
                                        ``stepScraped`` and the exact time             particle-boundary
                                        when the particle hits the                     interaction.
                                        boundary.                                      Saved in the boundary
                                                                                       buffers.
``n_x/y/z``            ``real``         Normal components to the boundary   SoA   RT   Added when there is
                                        on the position where the particle             particle-boundary
                                        hits the boundary.                             interaction.
                                                                                       Saved in the boundary
                                                                                       buffers.
``ionizationLevel``   ``int``           Ion ionization level                SoA   RT   Added when ionization
                                                                                       physics is used.
``opticalDepthQSR``   ``real``          QED: optical depth of the Quantum-  SoA   RT   Added when PICSAR QED
                                        Synchrotron process                            physics is used.
``opticalDepthBW``    ``real``          QED: optical depth of the Breit-    SoA   RT   Added when PICSAR QED
                                        Wheeler process                                physics is used.
====================  ================  ==================================  ===== ==== ======================

WarpX allows extra runtime attributes to be added to particle containers (through ``NewRealComp("attrname")`` or ``NewIntComp("attrname")``).
The attribute name can then be used to access the values of that attribute.
For example, using a particle iterator, ``pti``, to loop over the particles the command ``pti.GetAttribs(particle_comps["attrname"]).dataPtr();`` will return the values of the ``"attrname"`` attribute.

User-defined integer or real attributes are initialized when particles are generated in ``AddPlasma()``.
The attribute is initialized with a required user-defined parser function.
Please see the :ref:`input options <running-cpp-parameters-particle>` ``addIntegerAttributes`` and ``addRealAttributes`` for a user-facing documentation.

Commonly used runtime attributes are described in the table below and are all part of SoA particle storage:

==================  ================  =================================  ==============
Attribute name      ``int``/``real``  Description                        Default value
==================  ================  =================================  ==============
``prev_x/y/z``      ``real``          The coordinates of the particles   *user-defined*
                                      at the previous timestep.
``orig_x/y/z``      ``real``          The coordinates of the particles   *user-defined*
                                      when they were created.
==================  ================  =================================  ==============

A Python example that adds runtime options can be found in :download:`Examples/Tests/particle_data_python <../../../Examples/Tests/particle_data_python/inputs_test_2d_prev_positions_picmi.py>`

.. note::

   Only use ``_`` to separate components of vectors!
