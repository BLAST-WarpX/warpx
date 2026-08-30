# Copyright 2026 The WarpX Community
#
# License: BSD-3-Clause-LBNL

if(WarpX_LIB)
    add_executable(
        test_hybrid_electron_thermodynamics
        test_hybrid_electron_thermodynamics.cpp)
    target_link_libraries(
        test_hybrid_electron_thermodynamics
        PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_hybrid_electron_thermodynamics)
        target_compile_features(
            test_hybrid_electron_thermodynamics PRIVATE cuda_std_20)
        set_target_properties(
            test_hybrid_electron_thermodynamics PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON)
    else()
        target_compile_features(
            test_hybrid_electron_thermodynamics PRIVATE cxx_std_20)
        set_target_properties(
            test_hybrid_electron_thermodynamics PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON)
    endif()
    warpx_set_compile_warnings(test_hybrid_electron_thermodynamics)

    add_test(
        NAME test_hybrid_electron_thermodynamics_default
        COMMAND test_hybrid_electron_thermodynamics
            amrex.abort_on_unused_inputs=1)
    add_test(
        NAME test_hybrid_electron_thermodynamics_explicit_ideal_gas
        COMMAND test_hybrid_electron_thermodynamics
            hybrid_pic_model.electron_thermodynamics=ideal_gas
            amrex.abort_on_unused_inputs=1)
    add_test(
        NAME test_hybrid_electron_thermodynamics_fixed_charge_latent_energy
        COMMAND test_hybrid_electron_thermodynamics
            hybrid_pic_model.electron_thermodynamics=fixed_charge_latent_energy
            "hybrid_pic_model.electron_latent_transition_temperature_eV=10 100"
            "hybrid_pic_model.electron_latent_energy_eV=40 80"
            "hybrid_pic_model.electron_latent_sharpness=8 4"
            amrex.abort_on_unused_inputs=1)
    add_test(
        NAME test_hybrid_electron_thermodynamics_analytic_tungsten_composition
        COMMAND test_hybrid_electron_thermodynamics
            hybrid_pic_model.electron_thermodynamics=ideal_gas
            hybrid_pic_model.electron_composition_species=W
            hybrid_pic_model.electron_composition_atomic_mass=183.84
            hybrid_pic_model.electron_composition_atomic_number=74
            amrex.abort_on_unused_inputs=1)

    if(WarpX_SINGULARITY_EOS)
        add_executable(
            test_hybrid_electron_thermodynamics_spiner
            test_hybrid_electron_thermodynamics_spiner.cpp)
        target_link_libraries(
            test_hybrid_electron_thermodynamics_spiner
            PRIVATE lib_${WarpX_DIMS_LAST})
        target_compile_features(
            test_hybrid_electron_thermodynamics_spiner PRIVATE cxx_std_20)
        set_target_properties(
            test_hybrid_electron_thermodynamics_spiner PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON)
        warpx_set_compile_warnings(
            test_hybrid_electron_thermodynamics_spiner)
        if(WarpX_PRECISION STREQUAL "DOUBLE")
            add_test(
                NAME test_hybrid_electron_thermodynamics_spiner
                COMMAND test_hybrid_electron_thermodynamics_spiner)
        else()
            add_test(
                NAME test_hybrid_electron_thermodynamics_spiner_single_rejected
                COMMAND ${CMAKE_COMMAND}
                    "-DTEST_EXE=$<TARGET_FILE:test_hybrid_electron_thermodynamics>"
                    -P "${CMAKE_CURRENT_LIST_DIR}/expect_failure.cmake")
        endif()

        if(1 IN_LIST WarpX_DIMS
           AND WarpX_APP
           AND TARGET app_1d
           AND WarpX_PRECISION STREQUAL "DOUBLE"
           AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE"
           AND (WarpX_COMPUTE STREQUAL "OMP"
                OR WarpX_COMPUTE STREQUAL "NOACC"))
            set(spiner_z2_pressure_work_fixture_dir
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/electron_thermodynamics_sp5_z2_pressure_work_fixture")
            set(spiner_z2_pressure_work_fixture
                "${spiner_z2_pressure_work_fixture_dir}/electron_only_fixture.sp5")
            add_test(
                NAME electron_thermodynamics_sp5_z2_pressure_work_fixture
                COMMAND ${CMAKE_COMMAND} -E env
                    "WARPX_SPINER_FIXTURE_PATH=${spiner_z2_pressure_work_fixture}"
                    "WARPX_SPINER_FIXTURE_ATOMIC_MASS=13.019"
                    "WARPX_SPINER_FIXTURE_ATOMIC_NUMBER=2"
                    "WARPX_SPINER_FIXTURE_NUM_T=512"
                    "WARPX_SPINER_FIXTURE_RHO_MAX=15"
                    $<TARGET_FILE:test_hybrid_electron_thermodynamics_spiner>)
            set_tests_properties(
                electron_thermodynamics_sp5_z2_pressure_work_fixture
                PROPERTIES
                    FIXTURES_SETUP
                        electron_thermodynamics_sp5_z2_pressure_work_fixture)

            set(spiner_z2_pressure_work_analysis
                "analysis_test_1d_hybrid_spiner_z2_pressure_work.py diags/diag000000 diags/diag000001 diags/particle_energy.txt")

            add_warpx_test(
                test_1d_hybrid_spiner_z2_pressure_work_density_sentinel
                1
                1
                "inputs_test_1d_hybrid_spiner_z2_pressure_work_density_sentinel hybrid_pic_model.electron_eos_ions_table_file='${spiner_z2_pressure_work_fixture}'"
                "${spiner_z2_pressure_work_analysis}"
                OFF
                OFF)
            add_warpx_test(
                test_1d_hybrid_spiner_z2_pressure_work_density_sentinel_mpi
                1
                2
                "inputs_test_1d_hybrid_spiner_z2_pressure_work_density_sentinel_mpi hybrid_pic_model.electron_eos_ions_table_file='${spiner_z2_pressure_work_fixture}'"
                "${spiner_z2_pressure_work_analysis}"
                OFF
                OFF)

            foreach(spiner_z2_test IN ITEMS
                    test_1d_hybrid_spiner_z2_pressure_work_density_sentinel
                    test_1d_hybrid_spiner_z2_pressure_work_density_sentinel_mpi)
                if(TEST ${spiner_z2_test}.run)
                    set_tests_properties(
                        ${spiner_z2_test}.run
                        PROPERTIES
                            FIXTURES_REQUIRED
                                electron_thermodynamics_sp5_z2_pressure_work_fixture)
                endif()
            endforeach()
        endif()
    endif()

    if(1 IN_LIST WarpX_DIMS AND WarpX_APP AND TARGET app_1d)
        add_warpx_test(
            test_1d_electron_thermodynamics_mass_mismatch_rejected
            1
            1
            inputs_test_1d_electron_thermodynamics_mass_mismatch_rejected
            OFF
            OFF
            OFF)
        set_tests_properties(
            test_1d_electron_thermodynamics_mass_mismatch_rejected.run
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "Configured electron-thermodynamics atomic mass for species[^a-zA-Z0-9]*'ions'[^a-zA-Z0-9]*does not match particle mass[^a-zA-Z0-9]*effective-mass mappings")

        add_warpx_test(
            test_1d_electron_thermodynamics_mass_match_accepted
            1
            1
            inputs_test_1d_electron_thermodynamics_mass_match_accepted
            OFF
            OFF
            OFF)
    endif()
endif()
