.. _examples-ohm-solver-em-modes:

Ohm solver: Electromagnetic modes
=================================

In this example a simulation is seeded with a thermal plasma while an initial magnetic field is applied in either the
:math:`z` or :math:`x` direction. The simulation is progressed for a large number of steps and the resulting fields are
Fourier analyzed for Alfvén mode excitations.

Run
---

The same input script can be used for 1d, 2d or 3d Cartesian simulations as well
as replicating either the parallel propagating or ion-Bernstein modes as indicated below.

.. dropdown:: Script ``inputs_test_1d_ohm_solver_em_modes_picmi.py``

   .. literalinclude:: inputs_test_1d_ohm_solver_em_modes_picmi.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/ohm_solver_EM_modes/inputs_test_1d_ohm_solver_em_modes_picmi.py``.

For `MPI-parallel <https://www.mpi-forum.org>`__ runs, prefix these lines with ``mpiexec -n 4 ...`` or ``srun -n 4 ...``, depending on the system.

.. tab-set::

   .. tab-item:: Parallel propagating waves

      Execute:

      .. code-block:: bash

         python3 inputs_test_1d_ohm_solver_em_modes_picmi.py -dim {1/2/3} --bdir z

   .. tab-item:: Perpendicular propagating waves

      Execute:

      .. code-block:: bash

         python3 inputs_test_1d_ohm_solver_em_modes_picmi.py -dim {1/2/3} --bdir {x/y}

Analyze
-------

The following script reads the simulation output from the above example, performs
Fourier transforms of the field data and compares the calculated spectrum
to the theoretical dispersions.

.. dropdown:: Script ``analysis.py``

   .. literalinclude:: analysis.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/ohm_solver_EM_modes/analysis.py``.

Right and left circularly polarized electromagnetic waves are supported through the cyclotron motion of the ions, except
in a region of thermal resonances as indicated on the plot below.

.. figure:: https://user-images.githubusercontent.com/40245517/216207688-9c39374a-9e69-45b8-a588-35b087b83d27.png
   :alt: Parallel EM modes in thermal ion plasma
   :width: 70%

   Calculated Alvén waves spectrum with the theoretical dispersions overlaid.

Perpendicularly propagating modes are also supported, commonly referred to as ion-Bernstein modes.

.. figure:: https://user-images.githubusercontent.com/40245517/231217944-7d12b8d4-af4b-44f8-a1b9-a2b59ce3a1c2.png
   :alt: Perpendicular modes in thermal ion plasma
   :width: 50%

   Calculated ion Bernstein waves spectrum with the theoretical dispersion overlaid.

Ohm solver: Cylindrical normal modes
====================================

A RZ-geometry example case for normal modes propagating along an applied magnetic
field in a cylinder is also available. The analytical solution for these modes
are described in :cite:t:`ex-Stix1992` Chapter 6, Sec. 2.

Run
---

The following script initializes a thermal plasma in a metallic cylinder with
periodic boundaries at the cylinder ends.

.. dropdown:: Script ``inputs_test_rz_ohm_solver_em_modes_picmi.py``

   .. literalinclude:: inputs_test_rz_ohm_solver_em_modes_picmi.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/ohm_solver_EM_modes/inputs_test_rz_ohm_solver_em_modes_picmi.py``.

The example can be executed using:

.. code-block:: bash

   python3 inputs_test_rz_ohm_solver_em_modes_picmi.py

Analyze
-------

After the simulation completes the following script can be used to analyze the
field evolution and extract the normal mode dispersion relation. It performs a
standard Fourier transform along the cylinder axis and a Hankel transform in the
radial direction.

.. dropdown:: Script ``analysis_rz.py``

   .. literalinclude:: analysis_rz.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/ohm_solver_EM_modes/analysis_rz.py``.

The following figure was produced with the above analysis script, showing excellent
agreement between the calculated and theoretical dispersion relations.

.. figure:: https://user-images.githubusercontent.com/40245517/259251824-33e78375-81d8-410d-a147-3fa0498c66be.png
   :alt: Normal EM modes in a metallic cylinder
   :width: 90%

   Cylindrical normal mode dispersion comparing the calculated spectrum with the
   theoretical one.
