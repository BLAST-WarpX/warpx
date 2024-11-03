.. _examples-ohm-solver-ion-beam-instability:

Ohm solver: Ion Beam R Instability
==================================

In this example a low density ion beam interacts with a "core" plasma population which induces an instability.
Based on the relative density between the beam and the core plasma a resonant or non-resonant condition can
be accessed.

Run
---

The same input script can be used for 1d, 2d or 3d simulations as well as
replicating either the resonant or non-resonant condition as indicated below.

.. dropdown:: Script ``inputs_test_1d_ohm_solver_ion_beam_picmi.py``

   .. literalinclude:: inputs_test_1d_ohm_solver_ion_beam_picmi.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/ohm_solver_ion_beam_instability/inputs_test_1d_ohm_solver_ion_beam_picmi.py``.

For `MPI-parallel <https://www.mpi-forum.org>`__ runs, prefix these lines with ``mpiexec -n 4 ...`` or ``srun -n 4 ...``, depending on the system.

.. tab-set::

   .. tab-item:: Resonant case

      Execute:

      .. code-block:: bash

         python3 inputs_test_1d_ohm_solver_ion_beam_picmi.py -dim {1/2/3} --resonant

   .. tab-item:: Non-resonant case

      Execute:

      .. code-block:: bash

         python3 inputs_test_1d_ohm_solver_ion_beam_picmi.py -dim {1/2/3}

Analyze
-------

The following script reads the simulation output from the above example, performs
Fourier transforms of the field data and outputs the figures shown below.

.. dropdown:: Script ``analysis.py``

   .. literalinclude:: analysis.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/ohm_solver_ion_beam_instability/analysis.py``.

The figures below show the evolution of the y-component of the magnetic field as the beam and
core plasma interact.

.. figure:: https://user-images.githubusercontent.com/40245517/217923933-6bdb65cb-7d26-40d8-8687-7dd75274bd48.png
   :alt: Resonant ion beam R instability
   :width: 70%

.. figure:: https://user-images.githubusercontent.com/40245517/217925983-b91d6482-69bc-43c1-8c7d-23ebe7c69d49.png
   :alt: Non-resonant ion beam R instability
   :width: 70%

   Evolution of :math:`B_y` for resonant (top) and non-resonant (bottom) conditions.

The growth rates of the strongest growing modes for the resonant case are compared
to theory (dashed lines) in the figure below.

.. figure:: https://github.com/ECP-WarpX/WarpX/assets/40245517/a94bb6e5-30e9-4d8f-9e6b-844dc8f51d17
   :alt: Resonant ion beam R instability growth rates
   :width: 50%

   Time series of the mode amplitudes for m = 4, 5, 6 from simulation. The
   theoretical growth for these modes are also shown as dashed lines.
