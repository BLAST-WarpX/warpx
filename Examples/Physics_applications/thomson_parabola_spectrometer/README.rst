.. _examples-thomson_parabola_spectrometer:

Thomson Parabola Spectrometer
=============================

This example simulates a Thomson parabola spectrometer (TPS) :cite:t:`ex-Rhee1987`.

A TPS is a type of detector that separates incoming ions according to their charge-to-mass ratio (:math:`q/m`) and initial velocity (hence energy :math:`E_0 = 1/2 m v_0^2` if we assume non-relativistic dynamics).
TPSs are often used in laser-driven ion acceleration experiments, where different ion species are accelerated at once. To mimic this, we initialize a point-like source of 3 different ion species with different :math:`q/m` and :math:`E_0` (i.e. all ions have the same initial position, representative of a pinhole).

The ions propagate along :math:`z` through 4 subsequent regions:

 - a vacuum region, the distance between the pinhole and the TPS (0.1 m)
 - a region of constant electric field along :math:`x`, (0.19 m, 1e5 V/m)
 - a region of constant magnetic field along :math:`x`, (0.872 T, 0.12 m)
 - a vacuum region, the distance between the TPS and the screen of the detector (0.2 m)

The initial particle velocity :math:`v_0` is sampled from a uniform distribution in the range :math:`[v_{min}, v_{max}]` where :math:`v_{min} = \sqrt{E_{max}/m}`, :math:`v_{max} = \sqrt{2E_{max}/m}`, and :math:`E_{max}` is an input parameter for each species. We assume zero transverse momentum.

The ions are assumed to be test particles embedded in prescribed external fields, meaning that we neglect the self-field due to the ions' motion and the ions do not interact with each other.

The detector is modeled using a ``BoundaryScrapingDiagnostic`` at the upper :math:`z` boundary of the domain, which stores the attributes of the particles when they exit the simulation box from the corresponding edge. Note that the transverse box size is large enough such that all particles exit the domain from the upper :math:`z` side.

Run
---

The PICMI input file is not available for this example yet.

For `MPI-parallel <https://www.mpi-forum.org>`__ runs, prefix these lines with ``mpiexec -n 4 ...`` or ``srun -n 4 ...``, depending on the system.

.. literalinclude:: inputs_test_3d_thomson_parabola_spectrometer
   :language: ini
   :caption: You can copy this file from ``Examples/Physics_applications/thomson_parabola_spectrometer/inputs_test_3d_thomson_parabola_spectrometer``.

Visualize
---------

This figure below shows the ion trajectories starting from the pinhole (black star), entering the E and B field regions (purple box), up to the detector (gray plane).
The colors represent the different species: protons in blue, C :sup:`+4` in red, and C :sup:`+6` in green.
The particles are accelerated and deflected through the TPS.

.. figure:: https://gist.github.com/assets/17280419/3e45e5aa-d1fc-46e3-aa24-d9e0d6a74d1a
   :alt: Ion trajectories through a synthetic TPS.
   :width: 100%

In our simulation, the virtual detector stores all the particle data once entering it (i.e. exiting the simulation box).
The figure below shows the ions colored according to their species (same as above) and shaded according to their initial energy.
The :math:`x` coordinate represents the electric deflection, while :math:`y` the magnetic deflection.

.. figure:: https://gist.github.com/assets/17280419/4dd1adb7-b4ab-481d-bc24-8a7ca51471d9
   :alt: Synthetic TPS screen.
   :width: 100%

.. literalinclude:: analysis.py
   :language: ini
   :caption: You can copy this file from ``Examples/Physics_applications/thomson_parabola_spectrometer/analysis.py``.
