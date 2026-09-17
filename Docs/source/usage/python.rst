.. _usage-picmi:
.. _usage-picmi-run:

Inputs: PICMI Python Script
===========================

This documents on how to use WarpX as a Python script (e.g., ``python3 PICMI_script.py``).

WarpX uses the `PICMI standard <https://github.com/picmi-standard/picmi>`__ for its Python input files.
Complete example input files can be found in :ref:`the examples section <usage-examples>`.

.. tip::

   If you enjoy AI/LLM/agentic workflows, see our :ref:`AI (LLM)-Assisted Input File Design <ai_input_design>` section, too.

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

.. autopydantic_model:: pywarpx.picmi.Simulation
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.Cartesian3DGrid
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.Cartesian2DGrid
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.Cartesian1DGrid
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.CylindricalGrid
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.EmbeddedBoundary
    :inherited-members: BaseModel

Field solvers define the updates of electric and magnetic fields.

.. autopydantic_model:: pywarpx.picmi.ElectromagneticSolver
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.ElectrostaticSolver
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.HybridPICSolver
    :inherited-members: BaseModel

Object that allows smoothing of fields.

.. autopydantic_model:: pywarpx.picmi.BinomialSmoother
    :inherited-members: BaseModel

Evolve Schemes
--------------

These define the scheme use to evolve the fields and particles.
An instance of one of these would be passed as the `evolve_scheme` into the `Simulation`.

.. autopydantic_model:: pywarpx.picmi.ExplicitEvolveScheme
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.ThetaImplicitEMEvolveScheme
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.SemiImplicitEMEvolveScheme
    :inherited-members: BaseModel

There are several support classes use to specify components of the evolve schemes

.. autopydantic_model:: pywarpx.picmi.PicardNonlinearSolver
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.NewtonNonlinearSolver
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.GMRESLinearSolver
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.PETScKSPLinearSolver
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.CurlCurlMLMGPreconditioner
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.JacobiPreconditioner
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.PETScPreconditioner
    :inherited-members: BaseModel

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

.. autopydantic_model:: pywarpx.picmi.AnalyticInitialField
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.ConstantAppliedField
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.AnalyticAppliedField
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.LoadInitialField
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.PlasmaLens
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.Mirror
    :inherited-members: BaseModel

Diagnostics
-----------

.. autopydantic_model:: pywarpx.picmi.ParticleDiagnostic
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.ParticleBoundaryScrapingDiagnostic
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.FieldDiagnostic
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.TimeAveragedFieldDiagnostic
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.ElectrostaticFieldDiagnostic
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.Checkpoint
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.ReducedDiagnostic
    :inherited-members: BaseModel

Lab-frame diagnostics diagnostics are used when running boosted-frame simulations.

.. autopydantic_model:: pywarpx.picmi.LabFrameParticleDiagnostic
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.LabFrameFieldDiagnostic
    :inherited-members: BaseModel

Particles
---------

Species objects are a collection of particles with similar properties.
For instance, background plasma electrons, background plasma ions and an externally injected beam could each be their own particle species.

.. autopydantic_model:: pywarpx.picmi.Species
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.MultiSpecies
    :inherited-members: BaseModel

Particle distributions can be used for to initialize particles in a particle species.

.. autopydantic_model:: pywarpx.picmi.GaussianBunchDistribution
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.UniformDistribution
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.AnalyticDistribution
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.UniformFluxDistribution
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.AnalyticFluxDistribution
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.ParticleListDistribution
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.FromFileDistribution
    :inherited-members: BaseModel

Particle layouts determine how to microscopically place macro particles in a grid cell.

.. autopydantic_model:: pywarpx.picmi.GriddedLayout
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.PseudoRandomLayout
    :inherited-members: BaseModel

Other operations related to particles:

.. autopydantic_model:: pywarpx.picmi.CoulombCollisions
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.DSMCCollisions
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.MCCCollisions
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.FieldIonization
    :inherited-members: BaseModel

Laser Pulses
------------

Laser profiles can be used to initialize laser pulses in the simulation.

.. autopydantic_model:: pywarpx.picmi.GaussianLaser
    :inherited-members: BaseModel

.. autopydantic_model:: pywarpx.picmi.AnalyticLaser
    :inherited-members: BaseModel

Laser injectors control where to initialize laser pulses on the simulation grid.

.. autopydantic_model:: pywarpx.picmi.LaserAntenna
    :inherited-members: BaseModel
