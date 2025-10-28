.. _examples-free-electron-laser:

Free-electron laser
===================

This example shows how to simulate the physics of a free-electron laser (FEL) using WarpX.
In this example, a relativistic electron beam is sent through an undulator (represented by an external,
oscillating magnetic field). The radiation emitted by the beam grows exponentially
as the beam travels through the undulator, due to the Free-Electron-Laser instability.

The parameters of the simulation are taken from section 5.1 of :cite:t:`ex-Fallahi2020`.

The simulation is performed in 1D, and uses the boosted-frame technique as described in
:cite:t:`ex-VayFELA2009` and :cite:t:`ex-VayFELB2009` to reduce the computational cost (the Lorentz frame of the simulation is moving at the average speed of the beam in the undulator).
Even though the simulation is run in this boosted frame, the results are reconstructed in the
laboratory frame, using WarpX's ``BackTransformed`` diagnostic.

The effect of space-charge is intentionally turned off in this example, as it may not be properly modeled in 1D.
This is achieved by initializing two species of opposite charge (electrons and positrons) to
represent the physical electron beam, as discussed in :cite:t:`ex-VayFELB2009`.

Run
---

This example can be run with the WarpX executable using an input file: ``warpx.1d inputs_test_1d_fel``. For `MPI-parallel <https://www.mpi-forum.org>`__ runs, prefix these lines with ``mpiexec -n 4 ...`` or ``srun -n 4 ...``, depending on the system.

.. literalinclude:: inputs_test_1d_fel
   :language: ini
   :caption: You can copy this file from ``Examples/Physics_applications/free_electron_laser/inputs_test_1d_fel``.

Visualize
---------

The figure below shows the results of the simulation. The left panel shows the exponential growth of the radiation along the undulator (note that the vertical axis is plotted in log scale). The right panel shows a snapshot of the simulation,
1.6 m into the undulator. Microbunching of the beam is visible in the electron density (blue). One can also see the
emitted FEL radiation (red) slipping ahead of the beam.

.. figure:: https://gist.githubusercontent.com/RemiLehe/871a1e24c69e353c5dbb4625cd636cd1/raw/7f4e3da7e0001cff6c592190fee8622580bbe37a/FEL.png
   :alt: Results of the WarpX FEL simulation.
   :width: 100%

This figure was obtained with the script below, which can be run with ``python3 plot_sim.py``.

.. literalinclude:: plot_sim.py
   :language: ini
   :caption: You can copy this file from ``Examples/Physics_applications/free_electron_laser/plot_sim.py``.
