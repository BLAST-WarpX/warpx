.. _usage-examples:

Examples
========

This section allows you to **download input files** that correspond to different physical situations.

We provide two kinds of inputs:

* PICMI python input files, `with parameters described here <https://picmi-standard.github.io>`__.
* AMReX ``inputs`` files, :ref:`with parameters described here <running-cpp-parameters>`,

For a complete list of all example input files, also have a look at our `Examples/ <https://github.com/ECP-WarpX/WarpX/tree/development/Examples>`__ directory.
It contains folders and subfolders with self-describing names that you can try.
All these input files are automatically tested, so they should always be up-to-date.


Plasma-Based Acceleration
-------------------------

.. toctree::
   :maxdepth: 1

   examples/lwfa/README.rst
   examples/pwfa/README.rst
   pwfa.rst


Laser-Plasma Interaction
------------------------

.. toctree::
   :maxdepth: 1

   examples/laser_ion/README.rst
   examples/plasma_mirror/README.rst


Particle Accelerator & Beam Physics
-----------------------------------

.. toctree::
   :maxdepth: 1

   examples/gaussian_beam/README.rst
   examples/beam_beam_collision/README.rst
   examples/free_electron_laser/README.rst
   examples/thomson_parabola_spectrometer/README.rst

High Energy Astrophysical Plasma Physics
----------------------------------------

.. toctree::
   :maxdepth: 1

   examples/ohm_solver_magnetic_reconnection/README.rst


Microelectronics
----------------

`ARTEMIS (Adaptive mesh Refinement Time-domain ElectrodynaMIcs Solver) <https://ccse.lbl.gov/Research/Microelectronics/>`__ is based on WarpX and couples the Maxwell's equations implementation in WarpX with classical equations that describe quantum material behavior (such as, LLG equation for micromagnetics and London equation for superconducting materials) for quantifying the performance of `next-generation microelectronics <https://www.lbl.gov/research/microelectronics-and-beyond/>`__.

* `ARTEMIS examples <https://github.com/AMReX-Microelectronics/artemis/tree/development/Examples>`__
* `ARTEMIS manual <https://artemis-em.readthedocs.io>`__


Fundamental Plasma Physics
--------------------------

.. toctree::
   :maxdepth: 1

   examples/langmuir/README.rst
   examples/capacitive_discharge/README.rst


.. _examples-hybrid-model:

Kinetic-fluid Hybrid Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

WarpX includes a reduced plasma model in which electrons are treated as a massless
fluid while ions are kinetically evolved, and Ohm's law is used to calculate
the electric field. This model is appropriate for problems in which ion kinetics
dominate (ion cyclotron waves, for instance). See the
:ref:`theory section <theory-kinetic-fluid-hybrid-model>` for more details. Several
examples and benchmarks of this kinetic-fluid hybrid model are provided below.
A few of the examples are replications of the verification tests described in
:cite:t:`ex-MUNOZ2018`. The hybrid-PIC model was added to WarpX in
`PR #3665 <https://github.com/ECP-WarpX/WarpX/pull/3665>`_ - the figures in the
examples below were generated at that time.

.. toctree::
   :maxdepth: 1

   examples/ohm_solver_em_modes/README.rst
   examples/ohm_solver_ion_beam_instability/README.rst
   examples/ohm_solver_ion_Landau_damping/README.rst


High-Performance Computing and Numerics
---------------------------------------

The following examples are commonly used to study the performance of WarpX, e.g., for computing efficiency, scalability, and I/O patterns.
While all prior examples are used for such studies as well, the examples here need less explanation on the physics, less-detail tuning on load balancing, and often simply scale (weak or strong) by changing the number of cells, AMReX block size and number of compute units.

.. toctree::
   :maxdepth: 1

   examples/uniform_plasma/README.rst


Manipulating fields via Python
------------------------------

.. note::

   TODO: The section needs to be sorted into either science cases (above) or later sections (:ref:`workflows and Python API details <usage-python-extend>`).

An example of using Python to access the simulation charge density, solve the Poisson equation (using ``superLU``) and write the resulting electrostatic potential back to the simulation is given in the input file below. This example uses the ``fields.py`` module included in the ``pywarpx`` library.

* :download:`Direct Poisson solver example <../../../Examples/Physics_applications/capacitive_discharge/inputs_test_2d_background_mcc_picmi.py>`

An example of initializing the fields by accessing their data through Python, advancing the simulation for a chosen number of time steps, and plotting the fields again through Python. The simulation runs with 128 regular cells, 8 guard cells, and 10 PML cells, in each direction. Moreover, it uses div(E) and div(B) cleaning both in the regular grid and in the PML and initializes all available electromagnetic fields (E,B,F,G) identically.

* :download:`Unit pulse with PML <../../../Examples/Tests/python_wrappers/inputs_test_2d_python_wrappers_picmi.py>`


Many Further Examples, Demos and Tests
--------------------------------------

.. toctree::
   :maxdepth: 1

   examples/field_ionization/README.rst

WarpX runs over 200 integration tests on a variety of modeling cases, which validate and demonstrate its functionality.
Please see the `Examples/Tests/ <https://github.com/ECP-WarpX/WarpX/tree/development/Examples/Tests>`__ directory for many more examples.


Example References
------------------

.. bibliography::
    :keyprefix: ex-
