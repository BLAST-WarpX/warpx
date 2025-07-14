.. _examples-pierce-diode:

Pierce Diode at the Child–Langmuir Limit
===================

This example shows how to simulate the physics of a 1D Pierce diode operating
at the Child–Langmuir limit using WarpX. In this setup, an electron beam is injected
into a planar diode gap with a fixed potential difference, and the injected current density
is chosen to match the space-charge-limited current predicted by the Child–Langmuir law :cite:t:`ex-Zhang2017`.
Charged particles injected with initial velocity $v = 0$ will be accelerated toward the opposite
plate when the sign of the voltage difference is set appropriately.

Geometry
---

.. figure:: https://gist.githubusercontent.com/oshapoval/aaafd8d131c3e1ed0fefe348bc8db28b/raw/92c4089e1b9eb23ae258f60c386e38e04f9499a2/geometry_pierce_diode.png
   :alt: Two parallel conducting plates separated by the distance $d$ and powered by a
   voltage difference V will, for a proper value of $V$,
   accelerate charged particle initially at rest at one of the plates.
   :width: 100%

This test case provides a clear benchmark for validating the electrostatic solver’s ability
to reproduce the steady-state space-charge-limited flow between parallel electrods,
as predicded by the analytical Child–Langmuir law for 1D planar diodes.

The simulation is performed in 1D.
We assume two parallel conducting plates separated by the distance $d$, and with a constant potential difference $V$ between the two plates.
The electric potential and current density profiles,
obtained from the simulation can be directly compared to the analytical Child–Langmuir
expression for verification :cite:t:`ex-Zhang2020`. Assuming a steady state flow, the emitted current does not exceed a certain limit for
the potential $\phi(z)=V(\frac{z}{d})^{4/3}$ and the current density $J = \frac{4}{9} \varepsilon_9 \sqrt{\frac{2 \abs{q}}{m}} \frac{|V|^{3/2}}{d^2}$.
This limit in knows as the Child's Law or Child-Lamgmuir Law and give the maximum current that can be extracted for a given voltage and plate separation.

Run
---

This example can be run with the WarpX executable using an input file: ``warpx.1d inputs_test_1d_pierce_diode``. For `MPI-parallel <https://www.mpi-forum.org>`__ runs, prefix these lines with ``mpiexec -n 4 ...`` or ``srun -n 4 ...``, depending on the system.

.. literalinclude:: inputs_test_1d_pierce_diode
   :language: ini
   :caption: You can copy this file from ``Examples/Physics_applications/pierce_diode/inputs_test_1d_pierce_diode``.

Visualize
---------

The figure below shows the results of the simulation (orange cirves), which agrees well with the analytical Child–Langmuir law (black curves).

.. figure:: https://gist.githubusercontent.com/oshapoval/aaafd8d131c3e1ed0fefe348bc8db28b/raw/92c4089e1b9eb23ae258f60c386e38e04f9499a2/Pierce_Diode.png
   :alt: Results of the WarpX Pierce Diode simulation.
   :width: 100%

This figure was obtained with the script below, which can be run with ``python3 plot_sim.py``.

.. literalinclude:: plot_sim.py
   :language: ini
   :caption: You can copy this file from ``Examples/Physics_applications/pierce_diode/plot_sim.py``.
