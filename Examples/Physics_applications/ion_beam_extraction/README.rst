.. _examples-ion-beam-extraction:

Ion-Beam Extraction from a Plasma Source
========================================

This test simulates the extraction of a high-energy ion beam from a plasma source, using the same physical setup as in `this paper <https://pubs.aip.org/aip/rsi/article-abstract/81/2/02B108/1071759/Characterization-of-1-MW-40-keV-1-s-neutral-beam>`__.

The volume of the simulation box located at :math:`z<0` represents the plasma source, and is initially filled with a plasma made of positive Deuterium ions (:math:`D^{+}`) and electrons.
Some of the plasma ions are extracted and accelerated by electrodes (which are held at a fixed electrostatic potential), and form a continuous ion beam, reaching a final energy of approximately :math:`40\,\mathrm{keV}`.

The figure below shows a color map of the electrostatic potential (:math:`\phi`), along with black lines showing the position of the electrodes as well as red dots showing the
:math:`D^{+}` macroparticles. The bottom panel shows the kinetic energy of the extracted ion beam.

.. _ion_beam:

.. figure:: ion_beam_and_energy_1_beamlet.png
   :alt:  [fig:ion_beam] Color map of the electrostatic potential (:math:`\phi`) overlaid with contours of the embedded boundary (eb_covered field) and ion (:math:`D^{+}`) macroparticles, as well as kinetic energy of the extracted ion beam.
   :align: center

Plasma Source Setup
-------------------

To maintain the plasma density during the beam extraction process, additional ions and electrons are injected from the boundaries of the simulation box,
in the region :math:`z<0`. (Without this boundary injection, the plasma would progressively deplete as ions are accelerated out of the source region,
and both plasma ions and electrons with thermal motion are absorbed by the simulation boundaries.)
The flux with which ions and electrons are injected from the boundaries corresponds to that of a thermal plasma.

In the input script below, the plasma source thus consists of two parts:

1) At :math:`t=0`, initialization of the plasma in the volume corresponding to :math:`z<0`.

2) Throughout the simulation: continuous injection of ions and electrons from the simulation box boundaries located at :math:`z<0` -- from the :math:`\pm x`, :math:`\pm y` and  :math:`-z` boundaries.



Run
---

This example can be run with the WarpX executable using an input file: ``warpx.3d inputs_test_3d_ion_beam_extraction``.
For `MPI-parallel <https://www.mpi-forum.org>`__ runs, prefix these lines with ``mpiexec -n 4 ...`` or ``srun -n 4 ...``, depending on the system.
Note: For the CI test, we intentionally specified very high values for `self_fields_absolute_tolerance` and `self_fields_required_precision`, and lowered spatial resolution as well as number of particles per cell to make the test run faster. For production runs, feel free to lower or increase these values accordingly.

.. literalinclude:: inputs_test_3d_ion_beam_extraction
   :language: none
   :caption: You can copy this file from ``Examples/Physics_applications/ion_beam_extraction/inputs_test_3d_ion_beam_extraction``.

Visualize
---------

To visualize the results, you can use the provided plotting script, which reads the output diagnostics in openpmd format and generates plots of the electrostatic potential, ion beam macroparticles, as well as the ion beam energy distribution.
It also checks if the particle energies tail is within a relative tolerance of the target energy of :math:`40\,\mathrm{keV}`.
