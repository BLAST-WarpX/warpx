.. _examples-pierce-diode:

Pierce Diode at the Child–Langmuir Limit
========================================

This example shows how to simulate the physics of a 1D Pierce diode configuration operating
at the Child–Langmuir limit using WarpX. In this setup, an electron beam is injected
into a planar diode gap, consisting of by parallel conducting plates separated by the distance :math:`d` and powered by a voltage difference :math:`V` :numref:`fig_geom`.
The injected current density is chosen to match the space-charge-limited current predicted by the Child–Langmuir law :cite:t:`ex-Zhang2017`.
The law predicts the maximum current density that can flow between two parallel plates due to space-charge effects.
This test demonstrates that WarpX correctly reproduces the Child–Langmuir law for a given voltage and gap length.

Geometry
--------

The figure below schematically illustrates the problem geometry described above.

.. _fig_geom:

.. figure:: https://gist.githubusercontent.com/oshapoval/aaafd8d131c3e1ed0fefe348bc8db28b/raw/92c4089e1b9eb23ae258f60c386e38e04f9499a2/geometry_pierce_diode.png
   :alt:  [fig:geom] Two parallel conducting plates separated by the distance :math:`d` and powered by a voltage difference :math:`V`. Given that the two plates are parallel, here we simulate the problem in 1D with WarpX.
   :width: 80%
   :align: center

   Two parallel conducting plates separated by the distance :math:`d` and powered by a voltage difference :math:`V`. Given that the two plates are parallel, here we simulate the problem in 1D with WarpX.


Сhild–Langmuir Limit
--------------------

In steady state, the emitted current is limited by the Child–Langmuir law,
which defines the maximum current that can be transported across a planar diode for a given voltage and gap length :cite:t:`ex-Zhang2017`.
It can be shown that, at the Child-Langmuir limit (i.e. when this maximum current is reached), the potential and current density in the gap have the following expression:

.. math::
   \phi(z)=V\Big(\frac{z}{d}\Big)^{4/3},
   :label: child-langmuir-phi

.. math::
   J(z) = \frac{4}{9} \varepsilon_0 \sqrt{\frac{2 |q|}{m}} \frac{|V|^{3/2}}{d^2}.
   :label: child-langmuir-J

Run
---

This example can be run with the WarpX executable using an input file: ``warpx.1d inputs_test_1d_pierce_diode``.
For `MPI-parallel <https://www.mpi-forum.org>`__ runs, prefix these lines with ``mpiexec -n 4 ...`` or ``srun -n 4 ...``, depending on the system.

.. literalinclude:: inputs_test_1d_pierce_diode
   :language: ini
   :caption: You can copy this file from ``Examples/Physics_applications/pierce_diode/inputs_test_1d_pierce_diode``.

Visualize
---------

The figure below shows the results of the simulation (orange curves), which agrees well with the analytical Child–Langmuir law (black curves) (:eq:`child-langmuir-phi`, :eq:`child-langmuir-J`).

.. figure:: https://gist.githubusercontent.com/oshapoval/aaafd8d131c3e1ed0fefe348bc8db28b/raw/fc76b371d323dbca4e1c43b45055405ff1fc6de4/Pierce_Diode.png
   :alt: Results of the WarpX Pierce Diode simulation.
   :width: 100%

This figure was obtained with the script below, which can be run with ``python3 plot_sim.py``.

.. literalinclude:: plot_sim.py
   :language: ini
   :caption: You can copy this file from ``Examples/Physics_applications/pierce_diode/plot_sim.py``.
