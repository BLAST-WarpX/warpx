.. _theory:

Overview
========

.. _theory-pic:

In the *particle-in-cell method* :cite:p:`i-Birdsalllangdon`,
the electric and magnetic fields are solved on a discretized grid while particles are
evolved in continuous space. A high-level schematic of the method is shown in the
figure below with details of the different field solvers, particle handling algorithms,
additional physics modules, etc. described in the sections linked further down.

.. _fig-pic:

.. figure:: PIC.png
   :alt: [fig:PIC] The Particle-In-Cell (PIC) method follows the evolution of a collection of charged macro-particles (positively charged in blue on the left plot, negatively charged in red) that evolve self-consistently with their electromagnetic (or electrostatic) fields. The core PIC algorithm involves four operations at each time step: 1) evolve the velocity and position of the particles using the Newton-Lorentz equations, 2) deposit the charge and/or current densities through interpolation from the particles distributions onto the grid, 3) evolve Maxwell’s wave equations (for electromagnetic) or solve Poisson’s equation (for electrostatic) on the grid, 4) interpolate the fields from the grid onto the particles for the next particle push. Additional “add-ons” operations are inserted between these core operations to account for additional physics (e.g. absorption/emission of particles, addition of external forces to account for accelerator focusing or accelerating component) or numerical effects (e.g. smoothing/filtering of the charge/current densities and/or fields on the grid).

   The Particle-In-Cell (PIC) method follows the evolution of a collection of charged macro-particles (positively charged in blue on the left plot, negatively charged in red) that evolve self-consistently with their electromagnetic (or electrostatic) fields. The core PIC algorithm involves four operations at each time step: 1) evolve the velocity and position of the particles using the Newton-Lorentz equations, 2) deposit the charge and/or current densities through interpolation from the particles distributions onto the grid, 3) evolve Maxwell’s wave equations (for electromagnetic) or solve Poisson’s equation (for electrostatic) on the grid, 4) interpolate the fields from the grid onto the particles for the next particle push. Additional “add-ons” operations are inserted between these core operations to account for additional physics (e.g. absorption/emission of particles, addition of external forces to account for accelerator focusing or accelerating component) or numerical effects (e.g. smoothing/filtering of the charge/current densities and/or fields on the grid).

the most popular algorithm is the Particle-In-Cell (or PIC) technique,
which represents electromagnetic fields on a grid and particles by
a sample of macroparticles.

.. _theory-field_solvers:

Field Solvers
=============

.. toctree::
   :maxdepth: 1

   maxwell_solvers
   kinetic_fluid_hybrid_model

Grid & Geometries
=================

Boundary Conditions
===================

.. toctree::
   :maxdepth: 1

   boundary_conditions

Species Representations
=======================

.. toctree::
   :maxdepth: 1

   kinetic_particles
   cold_fluid_model

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
