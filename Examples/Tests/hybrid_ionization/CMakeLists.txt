# Copyright 2026 The WarpX Community
#
# License: BSD-3-Clause-LBNL

# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_hybrid_ionization_he  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_hybrid_ionization_he  # inputs
    "analysis.py --geometry cartesian --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (manufactured charge and energy ledgers are checked)
    OFF  # dependency
)

if(2 IN_LIST WarpX_DIMS AND WarpX_APP)
    set(invalid_gamma_test
        test_2d_hybrid_ionization_invalid_gamma_rejected.run)
    set(invalid_gamma_working_directory
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_2d_hybrid_ionization_invalid_gamma_rejected)
    file(MAKE_DIRECTORY ${invalid_gamma_working_directory})
    add_test(
        NAME ${invalid_gamma_test}
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:app_2d>
            -DTEST_INPUT=inputs_base_2d_hybrid_ionization_invalid_gamma_rejected
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake
    )
    set_tests_properties(
        ${invalid_gamma_test}
        PROPERTIES
            WORKING_DIRECTORY ${invalid_gamma_working_directory}
            ENVIRONMENT
                "AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/;OMP_NUM_THREADS=1"
    )
endif()

add_warpx_test(
    test_2d_hybrid_ionization_he_restart  # name
    2  # dims
    1  # nprocs
    "inputs_test_2d_hybrid_ionization_he_restart amr.restart='../test_2d_hybrid_ionization_he/diags/chk000001'"  # inputs
    "analysis.py --geometry cartesian --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --compare-reference ../test_2d_hybrid_ionization_he/diags/diag000002"  # analysis
    OFF  # checksum (analytic checks plus checkpoint/restart equivalence)
    test_2d_hybrid_ionization_he  # dependency
)

add_warpx_test(
    test_rcylinder_hybrid_ionization_he  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_hybrid_ionization_he  # inputs
    "analysis.py --geometry rcylinder --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (manufactured charge and energy ledgers are checked)
    OFF  # dependency
)
