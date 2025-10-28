.. _usage-picmi:
.. _usage-picmi-run:

Parameters: Python (PICMI)
==========================

This documents on how to use WarpX as a Python script (e.g., ``python3 PICMI_script.py``).

WarpX uses the `PICMI standard <https://github.com/picmi-standard/picmi>`__ for its Python input files.
Complete example input files can be found in :ref:`the examples section <usage-examples>`.

In the input file, instances of classes are created defining the various aspects of the simulation.
A variable of type :py:class:`pywarpx.picmi.Simulation` is the central object to which all other options are passed, defining the simulation time, field solver, registered species, etc.

Once the simulation is fully configured, it can be used in one of two modes.
**Interactive** use is the most common and can be :ref:`extended with custom runtime functionality <usage-python-extend>`:

.. tab-set::

   .. tab-item:: Interactive

      :py:meth:`~pywarpx.picmi.Simulation.step`: run directly from Python

   .. tab-item:: Preprocessor

      :py:meth:`~pywarpx.picmi.Simulation.write_input_file`: create an :ref:`inputs file for a WarpX executable <running-cpp-parameters>`

When run directly from Python, one can also extend WarpX with further custom user logic.
See the :ref:`detailed workflow page <usage-python-extend>` on how to extend WarpX from Python.


.. _usage-picmi-parameters:

Simulation and Grid Setup
-------------------------

.. autoclass:: pywarpx.picmi.Simulation
    :members: step, add_species, add_laser, add_applied_field, write_input_file

.. autoclass:: pywarpx.picmi.Cartesian3DGrid

.. autoclass:: pywarpx.picmi.Cartesian2DGrid

.. autoclass:: pywarpx.picmi.Cartesian1DGrid

.. autoclass:: pywarpx.picmi.CylindricalGrid

.. autoclass:: pywarpx.picmi.EmbeddedBoundary

Field solvers define the updates of electric and magnetic fields.

.. autoclass:: pywarpx.picmi.ElectromagneticSolver

.. autoclass:: pywarpx.picmi.ElectrostaticSolver

Object that allows smoothing of fields.

.. autoclass:: pywarpx.picmi.BinomialSmoother

Evolve Schemes
--------------

These define the scheme use to evolve the fields and particles.
An instance of one of these would be passed as the `evolve_scheme` into the `Simulation`.

.. autoclass:: pywarpx.picmi.ExplicitEvolveScheme

.. autoclass:: pywarpx.picmi.ThetaImplicitEMEvolveScheme

.. autoclass:: pywarpx.picmi.SemiImplicitEMEvolveScheme

There are several support classes use to specify components of the evolve schemes

.. autoclass:: pywarpx.picmi.PicardNonlinearSolver

.. autoclass:: pywarpx.picmi.NewtonNonlinearSolver

.. autoclass:: pywarpx.picmi.GMRESLinearSolver

Constants
---------

For convenience, the PICMI interface defines the following constants,
which can be used directly inside any PICMI script. The values are in SI units.

- ``picmi.constants.c``: The speed of light in vacuum.
- ``picmi.constants.ep0``: The vacuum permittivity :math:`\epsilon_0`
- ``picmi.constants.mu0``: The vacuum permeability :math:`\mu_0`
- ``picmi.constants.q_e``: The elementary charge (absolute value of the charge of an electron).
- ``picmi.constants.m_e``: The electron mass
- ``picmi.constants.m_p``: The proton mass

Applied fields
--------------

Instances of the classes below need to be passed to the method `add_applied_field` of the `Simulation` class.

.. autoclass:: pywarpx.picmi.AnalyticInitialField

.. autoclass:: pywarpx.picmi.ConstantAppliedField

.. autoclass:: pywarpx.picmi.AnalyticAppliedField

.. autoclass:: pywarpx.picmi.LoadInitialField

.. autoclass:: pywarpx.picmi.PlasmaLens

.. autoclass:: pywarpx.picmi.Mirror

Diagnostics
-----------

.. autoclass:: pywarpx.picmi.ParticleDiagnostic

.. autoclass:: pywarpx.picmi.FieldDiagnostic

.. autoclass:: pywarpx.picmi.TimeAveragedFieldDiagnostic

.. autoclass:: pywarpx.picmi.ElectrostaticFieldDiagnostic

.. autoclass:: pywarpx.picmi.Checkpoint

.. autoclass:: pywarpx.picmi.ReducedDiagnostic

Lab-frame diagnostics diagnostics are used when running boosted-frame simulations.

.. autoclass:: pywarpx.picmi.LabFrameParticleDiagnostic

.. autoclass:: pywarpx.picmi.LabFrameFieldDiagnostic

Particles
---------

Species objects are a collection of particles with similar properties.
For instance, background plasma electrons, background plasma ions and an externally injected beam could each be their own particle species.

.. autoclass:: pywarpx.picmi.Species

.. autoclass:: pywarpx.picmi.MultiSpecies

Particle distributions can be used for to initialize particles in a particle species.

.. autoclass:: pywarpx.picmi.GaussianBunchDistribution

.. autoclass:: pywarpx.picmi.UniformDistribution

.. autoclass:: pywarpx.picmi.AnalyticDistribution

.. autoclass:: pywarpx.picmi.UniformFluxDistribution

.. autoclass:: pywarpx.picmi.AnalyticFluxDistribution

.. autoclass:: pywarpx.picmi.ParticleListDistribution

Particle layouts determine how to microscopically place macro particles in a grid cell.

.. autoclass:: pywarpx.picmi.GriddedLayout

.. autoclass:: pywarpx.picmi.PseudoRandomLayout

Other operations related to particles:

.. autoclass:: pywarpx.picmi.CoulombCollisions

.. autoclass:: pywarpx.picmi.DSMCCollisions

.. autoclass:: pywarpx.picmi.MCCCollisions

.. autoclass:: pywarpx.picmi.FieldIonization

Laser Pulses
------------

Laser profiles can be used to initialize laser pulses in the simulation.

.. autoclass:: pywarpx.picmi.GaussianLaser

.. autoclass:: pywarpx.picmi.AnalyticLaser

Laser injectors control where to initialize laser pulses on the simulation grid.

.. autoclass:: pywarpx.picmi.LaserAntenna
