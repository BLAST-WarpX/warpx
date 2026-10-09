Nuclear fusion tests
====================

Anisotropic D-D and D-T beam-target fusion
-------------------------------------------

The automated anisotropic beam-target tests exercise the energy-dependent angular distribution of fusion products at the deuterium beam momentum ``beta*gamma = 0.1``.
The two additional momenta ``0.01`` and ``0.05`` can be run locally for both reactions to reproduce the quantities plotted in Figure 2 of `van de Wetering et al. (2025) <https://doi.org/10.1103/zwjx-jbxl>`__.

Each CTest analysis validates its run independently using integral neutron spectrum observables and kinematic consistency checks.
The tests use 20,000 particles per cell to reduce their runtime.
For a closer match to the benchmark figure in the paper, use 40,000 particles per cell for both reactant species in the D-D and D-T base input files.

From the WarpX source directory, run all six simulations directly with a 3D MPI-enabled WarpX executable.
The following commands use each reaction's 10% input file as a template and override ``deuterium_beam.momentum_function_uz`` on the command line.
They use the same two MPI ranks and runtime parameters as the automated tests and replace any existing output directories with the same names under ``build/bin``:

.. code-block:: bash

   # Reproduce the anisotropic D-D and D-T fusion plots in Figure 2 of the PRE paper.
   # Run from the WarpX root directory after building a 3D MPI-enabled executable.
   warpx_dir=$(pwd)
   test_dir="${warpx_dir}/Examples/Tests/nuclear_fusion"
   analysis_script="${test_dir}/analysis_fusion_anisotropic_beam_target.py"
   warpx_executable=$(find "${warpx_dir}/build/bin" -maxdepth 1 -type f -name 'warpx.3d*' -executable -print -quit)

   for reaction in deuterium_deuterium deuterium_tritium; do
     diag_dirs=()
     for momentum in 1 5 10; do
       test_name="test_3d_${reaction}_fusion_anisotropic_beam_target_uz_${momentum}pct"
       input_name="inputs_test_3d_${reaction}_fusion_anisotropic_beam_target_uz_10pct"
       printf -v beam_momentum '0.%02d' "${momentum}"
       run_dir="${warpx_dir}/build/bin/${test_name}"
       cmake -E remove_directory "${run_dir}"
       cmake -E make_directory "${run_dir}"
       (
         cd "${run_dir}"
         OMP_NUM_THREADS=1 AMREX_INPUTS_FILE_PREFIX="${test_dir}/" \
           mpiexec -n 2 "${warpx_executable}" "${input_name}" \
           "deuterium_beam.momentum_function_uz(x,y,z)=${beam_momentum}" \
           amrex.abort_on_unused_inputs=1 amrex.throw_exception=1 amrex.the_arena_init_size=0 \
           warpx.always_warn_immediately=1 warpx.do_dynamic_scheduling=0 warpx.serialize_initial_conditions=1
       )
       diag_dirs+=("${run_dir}/diags/diag1")
     done
     python "${analysis_script}" --plot "${diag_dirs[@]}"
   done

The script writes ``deuterium_deuterium_fusion_anisotropic_beam_target_neutron_spectrum.png`` and ``deuterium_tritium_fusion_anisotropic_beam_target_neutron_spectrum.png`` in the current directory.
Solid curves show the normalized neutron energy spectra; dashed curves show the weighted mean center-of-momentum-frame emission angle in each energy bin.
