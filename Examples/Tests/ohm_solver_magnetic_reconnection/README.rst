.. _examples-ohm-solver-magnetic-reconnection:

Ohm Solver: Magnetic Reconnection
=================================

Hybrid-PIC codes are often used to simulate magnetic reconnection in space plasmas.
An example of magnetic reconnection from a force-free sheet is provided, based on
the simulation described in :cite:t:`ex-Le2016`.

Run
---

The following **Python** script configures and launches the simulation.

.. dropdown:: Script ``inputs_test_2d_ohm_solver_magnetic_reconnection_picmi.py``

   .. literalinclude:: inputs_test_2d_ohm_solver_magnetic_reconnection_picmi.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/ohm_solver_magnetic_reconnection/inputs_test_2d_ohm_solver_magnetic_reconnection_picmi.py``.

Running the full simulation should take about 4 hours if executed on 1 V100 GPU.
For `MPI-parallel <https://www.mpi-forum.org>`__ runs, prefix these lines with
``mpiexec -n 4 ...`` or ``srun -n 4 ...``, depending on the system.

   .. code-block:: bash

      python3 inputs_test_2d_ohm_solver_magnetic_reconnection_picmi.py

Analyze
-------

The following script extracts the reconnection rate as a function of time and
animates the evolution of the magnetic field (as shown below).

.. dropdown:: Script ``analysis.py``

   .. literalinclude:: analysis.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/ohm_solver_magnetic_reconnection/analysis.py``.

.. figure:: https://user-images.githubusercontent.com/40245517/229639784-b5d3b596-3550-4570-8761-8d9a67aa4b3b.gif
   :alt: Magnetic reconnection.
   :width: 70%

   Magnetic reconnection from a force-free sheet.
