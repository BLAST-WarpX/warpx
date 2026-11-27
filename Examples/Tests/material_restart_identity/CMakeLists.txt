# Copyright 2026 The WarpX Community
#
# License: BSD-3-Clause-LBNL

if(WarpX_LIB
   AND WarpX_SINGULARITY_EOS
   AND WarpX_MATERIAL_OPACITY_HDF5)
    add_executable(
        test_material_restart_identity
        test_material_restart_identity.cpp)
    target_link_libraries(
        test_material_restart_identity
        PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_material_restart_identity)
        target_compile_features(test_material_restart_identity PRIVATE cuda_std_20)
    endif()
    target_compile_features(
        test_material_restart_identity PRIVATE cxx_std_20)
    set_target_properties(
        test_material_restart_identity PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_STANDARD_REQUIRED ON)
    warpx_set_compile_warnings(test_material_restart_identity)

    function(add_material_identity_test name expected_failure)
        add_test(
            NAME material_restart_identity_${name}
            COMMAND ${CMAKE_COMMAND} -E env
                "MATERIAL_RESTART_IDENTITY_CASE=${name}"
                "OMP_NUM_THREADS=1"
                $<TARGET_FILE:test_material_restart_identity>)
        set_tests_properties(
            material_restart_identity_${name}
            PROPERTIES
                TIMEOUT 30
                WILL_FAIL ${expected_failure})
    endfunction()

    add_material_identity_test(unchanged FALSE)
    add_material_identity_test(relocated FALSE)
    add_material_identity_test(changed_content TRUE)
    add_material_identity_test(changed_binding TRUE)
    add_material_identity_test(changed_material_id TRUE)
    add_material_identity_test(missing TRUE)
    add_material_identity_test(invalid_manifest TRUE)
    add_material_identity_test(legacy_rejected TRUE)

    add_test(
        NAME material_restart_identity_legacy_override
        COMMAND ${CMAKE_COMMAND} -E env
            "MATERIAL_RESTART_IDENTITY_CASE=legacy"
            "OMP_NUM_THREADS=1"
            $<TARGET_FILE:test_material_restart_identity>
            materials.restart_allow_legacy_tables=1)
    set_tests_properties(
        material_restart_identity_legacy_override
        PROPERTIES TIMEOUT 30)

    if(WarpX_MPI AND MPIEXEC_EXECUTABLE)
        add_test(
            NAME material_restart_identity_mpi_agreement
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPI_ALLOW_ROOT}
                ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_POSTFLAGS}
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_material_restart_identity>)
        add_test(
            NAME material_restart_identity_mpi_one_rank_failure
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPI_ALLOW_ROOT}
                ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_POSTFLAGS}
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_material_restart_identity>)
        set_tests_properties(
            material_restart_identity_mpi_agreement
            material_restart_identity_mpi_one_rank_failure
            PROPERTIES
                PROCESSORS 2
                TIMEOUT 30
                ENVIRONMENT
                    "MATERIAL_RESTART_IDENTITY_CASE=mpi_agree;OMP_NUM_THREADS=1")
        set_tests_properties(
            material_restart_identity_mpi_one_rank_failure
            PROPERTIES
                ENVIRONMENT
                    "MATERIAL_RESTART_IDENTITY_CASE=mpi_failure;OMP_NUM_THREADS=1")
        set_tests_properties(
            material_restart_identity_mpi_one_rank_failure
            PROPERTIES WILL_FAIL TRUE)
    endif()
endif()
