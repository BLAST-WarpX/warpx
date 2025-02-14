.. _multiphysics-qed:

Quantum Electrodynamics (QED)
=============================

Quantum synchrotron
-------------------

.. note::

   Section empty!

Breit-Wheeler
-------------

.. note::

   Section empty!

Schwinger process
-----------------

If the code is compiled with QED and the user activates the Schwinger process in the input file,
electron-positron pairs can be created in vacuum in the function
``MultiParticleContainer::doQEDSchwinger``:

.. doxygenfunction:: MultiParticleContainer::doQEDSchwinger

``MultiParticleContainer::doQEDSchwinger`` in turn calls the function ``filterCreateTransformFromFAB``:

Filter Create Transform Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``filterCreateTransformFromFAB`` proceeds in three steps.
In the filter phase, we loop on every cell and calculate the number of physical pairs created within
the time step dt as a function of the electromagnetic field at the given cell position.
This probabilistic calculation is done via a wrapper that calls the ``PICSAR`` library.
In the create phase, the particles are created at the desired positions, currently at the cell nodes.
In the transform phase, we assign a weight to the particles depending on the number of physical
pairs created.
At most one macroparticle is created per cell per timestep per species, with a weight corresponding to
the total number of physical pairs created.

So far the Schwinger module requires using ``warpx.grid_type = collocated`` or
``algo.field_gathering = momentum-conserving`` (so that the auxiliary fields are calculated on the nodes)
and is not compatible with either mesh refinement, RZ coordinates or single precision.
