# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
function(register_native_rz_nodal_tests)
    if(NOT WarpX_LIB OR NOT RZ IN_LIST WarpX_DIMS OR
       NOT WarpX_PRECISION STREQUAL DOUBLE OR
       NOT WarpX_PARTICLE_PRECISION STREQUAL DOUBLE)
        return()
    endif()
    add_executable(test_rz_native_nodal_curl test_rz_native_nodal_curl.cpp)
    target_link_libraries(test_rz_native_nodal_curl PRIVATE lib_rz)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_rz_native_nodal_curl)
        target_compile_features(test_rz_native_nodal_curl PRIVATE cuda_std_20)
    else()
        target_compile_features(test_rz_native_nodal_curl PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_rz_native_nodal_curl)
    add_warpx_test(test_rz_native_magnetic_diffusion RZ 1
        inputs_rz_native_magnetic_diffusion
        "analysis_rz_native_magnetic_diffusion.py --run-dir ." OFF OFF)
    add_warpx_test(test_rz_native_magnetic_diffusion_refined RZ 1
        "inputs_rz_native_magnetic_diffusion amr.n_cell=128 8 warpx.const_dt=1.25e-4 max_step=400 diag.intervals=400"
        "analysis_rz_native_magnetic_diffusion.py --run-dir . --coarse-reference ../test_rz_native_magnetic_diffusion"
        OFF test_rz_native_magnetic_diffusion)
    if(WarpX_MPI)
        add_warpx_test(test_rz_native_magnetic_diffusion_mpi RZ 2
            inputs_rz_native_magnetic_diffusion
            "analysis_rz_native_magnetic_diffusion.py --run-dir ." OFF OFF)
    endif()
    foreach(variant IN ITEMS serial refined mpi)
        if(variant STREQUAL mpi AND NOT WarpX_MPI)
            continue()
        endif()
        set(name test_rz_native_nodal_curl_${variant})
        set(directory "${CMAKE_CURRENT_BINARY_DIR}/${name}")
        file(MAKE_DIRECTORY "${directory}")
        set(launcher)
        set(args)
        if(variant STREQUAL mpi)
            set(launcher ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS})
        elseif(variant STREQUAL refined)
            list(APPEND args "test.refine=64")
        endif()
        add_test(NAME ${name} COMMAND ${launcher} $<TARGET_FILE:test_rz_native_nodal_curl>
            ${MPIEXEC_POSTFLAGS} "${CMAKE_CURRENT_SOURCE_DIR}/inputs_rz_native_nodal_curl"
            "amrex.abort_on_unused_inputs=1" "amrex.throw_exception=1" ${args})
        set_tests_properties(${name} PROPERTIES WORKING_DIRECTORY "${directory}"
            ENVIRONMENT "OMP_NUM_THREADS=2")
        if(variant STREQUAL mpi)
            set_tests_properties(${name} PROPERTIES PROCESSORS 2 RUN_SERIAL TRUE
                ENVIRONMENT "OMP_NUM_THREADS=1")
        endif()
    endforeach()
endfunction()
register_native_rz_nodal_tests()
