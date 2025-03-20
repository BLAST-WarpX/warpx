.. _examples-ohm-solver-ion-landau-damping:

Ohm solver: Ion Landau Damping
==============================

Landau damping is a well known process in which electrostatic (acoustic) waves are
damped by transferring energy to particles satisfying a resonance condition.
The process can be simulated by seeding a plasma with a specific acoustic mode
(density perturbation) and tracking the strength of the mode as a function of time.

Run
---

The same input script can be used for 1d, 2d or 3d simulations and to sweep different
temperature ratios.

.. dropdown:: Script ``inputs_test_2d_ohm_solver_landau_damping_picmi.py``

   .. literalinclude:: inputs_test_2d_ohm_solver_landau_damping_picmi.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/ohm_solver_ion_Landau_damping/inputs_test_2d_ohm_solver_landau_damping_picmi.py``.

For `MPI-parallel <https://www.mpi-forum.org>`__ runs, prefix these lines with ``mpiexec -n 4 ...`` or ``srun -n 4 ...``, depending on the system.

   .. code-block:: bash

      python3 inputs_test_2d_ohm_solver_landau_damping_picmi.py -dim {1/2/3} --temp_ratio {value}

Analyze
-------

The following script extracts the amplitude of the seeded mode as a function
of time and compares it to the theoretical damping rate.

.. dropdown:: Script ``analysis.py``

   .. literalinclude:: analysis.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/ohm_solver_ion_Landau_damping/analysis.py``.

The figure below shows a set of such simulations with parameters matching those
described in section 4.5 of :cite:t:`ex-MUNOZ2018`.
The straight lines show the theoretical damping rate for the given temperature ratios.

.. figure:: https://user-images.githubusercontent.com/40245517/230523935-3c8d63bd-ee69-4639-b111-f06dad5587f6.png
   :alt: Ion Landau damping
   :width: 70%

   Decay of seeded modes as a function of time for different electron-ion temperature ratios.
   The theoretical damping of the given modes are shown in dashed lines.
