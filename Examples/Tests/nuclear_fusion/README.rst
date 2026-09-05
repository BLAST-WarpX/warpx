Nuclear fusion tests
====================

Anisotropic D-D and D-T beam-target fusion
-------------------------------------------

The anisotropic beam-target tests exercise the energy-dependent angular
distribution of fusion products at three deuterium beam momenta,
``beta*gamma = 0.01``, ``0.05``, and ``0.1``.  Together, the three runs for
each reaction reproduce the quantities plotted in Figure 2 of
`van de Wetering et al. (2025) <https://doi.org/10.1103/zwjx-jbxl>`__.

Each CTest analysis validates its run independently using integral neutron
spectrum observables. The tests use 20,000 particles per cell to
reduce their runtime. For a closer match to the benchmark figure in the
paper, use 40,000 particles per cell for both reactant species in the D-D and
D-T base input files.

To generate the comparison plots, first run the six simulations without their
analysis or cleanup tests:

.. code-block:: bash

   ctest --test-dir build --output-on-failure \
     -R "test_3d_deuterium_(deuterium|tritium)_fusion_anisotropic_beam_target_.*pc\\.run$"

Then, from the WarpX source directory, combine the three diagnostic outputs
for each reaction:

.. code-block:: bash

   python Examples/Tests/nuclear_fusion/analysis_fusion_anisotropic_beam_target.py \
     --plot \
     build/bin/test_3d_deuterium_deuterium_fusion_anisotropic_beam_target_1pc/diags/diag1 \
     build/bin/test_3d_deuterium_deuterium_fusion_anisotropic_beam_target_5pc/diags/diag1 \
     build/bin/test_3d_deuterium_deuterium_fusion_anisotropic_beam_target_10pc/diags/diag1

   python Examples/Tests/nuclear_fusion/analysis_fusion_anisotropic_beam_target.py \
     --plot \
     build/bin/test_3d_deuterium_tritium_fusion_anisotropic_beam_target_1pc/diags/diag1 \
     build/bin/test_3d_deuterium_tritium_fusion_anisotropic_beam_target_5pc/diags/diag1 \
     build/bin/test_3d_deuterium_tritium_fusion_anisotropic_beam_target_10pc/diags/diag1

The commands write ``deuterium_deuterium_fusion_anisotropic_beam_target_neutron_spectrum.png`` and
``deuterium_tritium_fusion_anisotropic_beam_target_neutron_spectrum.png`` in the current directory.  Solid
curves show the normalized neutron energy spectra; dashed curves show the
weighted mean laboratory-frame emission angle in each energy bin.
