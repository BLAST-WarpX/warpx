.. _theory:

Overview
========

.. _theory-pic:

WarpX simulates the **self-consistent** evolution of particle species (e.g., electrons, ions, etc.) in the presence of electric and magnetic fields.
In this context, *self-consistent* indicates that the particle dynamics are influenced by the fields, while the fields themselves evolve in response to the particles' changing charge and current density.

The fields are represented on a discrete spatial grid (see :ref:`theory-grid`).
The species are most commonly represented by discrete macro-particles moving continuously through the grid, but can also be represented as fluids in WarpX (see :ref:`theory-species_representations`).

At each time step of a simulation, both the species and the fields are updated -- using the equations of motion and field equation respectively.
Different types of field equations can be used in WarpX (see :ref:`theory-field_solvers`), and this choice determines many of the algorithmic details --
such as the maximum allowed time-step, the time staggering of the fields and particles position/momentum, the exact time-stepping algorithm, and whether the species' charge density or current density is the source of the fields' update.

.. _fig-pic:

.. figure:: PIC.png
   :alt: figure not found

   The core Particle-In-Cell (PIC) algorithm involves four operations at each time step: 1) evolve the field equation on the grid, 4) interpolate the fields from the grid onto the particles for the next particle push.

.. _theory-field_solvers:

Field Solvers
=============

.. toctree::
   :maxdepth: 1

   maxwell_solvers
   kinetic_fluid_hybrid_model

.. _theory-grid:

Grid & Geometries
=================

Species Representations
=======================

.. toctree::
   :maxdepth: 1

   kinetic_particles
   cold_fluid_model

Boundary Conditions
===================

.. toctree::
   :maxdepth: 1

   boundary_conditions

.. _theory-species_representations:

Multiphysics Processes
======================

.. toctree::
   :maxdepth: 1

   multiphysics_extensions

Advanced Modes of Running
=========================

.. toctree::
   :maxdepth: 1

   amr
   boosted_frame

.. bibliography::
    :keyprefix: i-
