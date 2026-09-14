# Copyright 2026 The WarpX Community
#
# License: BSD-3-Clause-LBNL

if(WarpX_LIB AND RZ IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL DOUBLE)
    add_executable(test_rz_shape_integral test_shape_integral.cpp)
    target_link_libraries(test_rz_shape_integral PRIVATE lib_rz)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_rz_shape_integral)
    endif()
    warpx_set_compile_warnings(test_rz_shape_integral)
    add_test(NAME test_rz_shape_integral COMMAND test_rz_shape_integral
        amrex.the_arena_init_size=0)
    set_tests_properties(test_rz_shape_integral PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
    if(WarpX_PARTICLE_PRECISION STREQUAL DOUBLE)
        add_warpx_test(
            test_rz_hybrid_shape_tail
            RZ
            1
            inputs_base_rz_shape_tail
            analysis_shape_tail.py
            OFF
            OFF)
        add_warpx_test(
            test_rz_implicit_vacuum_crossing
            RZ
            1
            inputs_base_rz_implicit_vacuum
            analysis_implicit_vacuum.py
            OFF
            OFF)
        if(WarpX_MPI)
            add_warpx_test(
                test_rz_implicit_vacuum_crossing_mpi
                RZ
                2
                inputs_base_rz_implicit_vacuum
                analysis_implicit_vacuum.py
                OFF
                OFF)
        endif()
    endif()
    add_executable(test_rz_implicit_charge_energy test_implicit_charge_energy.cpp)
    target_link_libraries(test_rz_implicit_charge_energy PRIVATE lib_rz)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_rz_implicit_charge_energy)
    endif()
    warpx_set_compile_warnings(test_rz_implicit_charge_energy)
    add_test(NAME test_rz_implicit_charge_energy COMMAND test_rz_implicit_charge_energy
        amrex.the_arena_init_size=0)
    set_tests_properties(test_rz_implicit_charge_energy PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
    if(NOT Python_EXECUTABLE)
        find_package(Python COMPONENTS Interpreter REQUIRED)
    endif()
    add_test(NAME test_rz_implicit_metric_rejected COMMAND ${Python_EXECUTABLE}
        ${WarpX_SOURCE_DIR}/Examples/Tests/implicit/check_command_output.py
        --expect-failure --expect "Implicit charge-energy remap violated" --
        $<TARGET_FILE:test_rz_implicit_charge_energy>
        amrex.the_arena_init_size=0 test.corrupt_metric=1)
    set_tests_properties(test_rz_implicit_metric_rejected PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
    if(WarpX_MPI)
        add_test(NAME test_rz_implicit_charge_energy_mpi
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPI_ALLOW_ROOT} ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_POSTFLAGS} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_rz_implicit_charge_energy> amrex.the_arena_init_size=0)
        set_tests_properties(test_rz_implicit_charge_energy_mpi PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=1" PROCESSORS 2)
    endif()
endif()

if(WarpX_LIB)
foreach(dimension IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(suffix ${dimension})
    set(offset_test test_${suffix}_temperature_deposition_offset)
    add_executable(${offset_test} test_temperature_offset.cpp)
    target_link_libraries(${offset_test} PRIVATE lib_${suffix})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(${offset_test})
    endif()
    warpx_set_compile_warnings(${offset_test})
    add_test(NAME ${offset_test} COMMAND ${offset_test} amrex.the_arena_init_size=0)
    set_tests_properties(${offset_test} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
endforeach()
endif()

if(WarpX_LIB)
    add_executable(
        test_hybrid_metric_transport
        test_hybrid_metric_transport.cpp)
    target_link_libraries(
        test_hybrid_metric_transport
        PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_hybrid_metric_transport)
        target_compile_features(
            test_hybrid_metric_transport PRIVATE cuda_std_20)
        set_target_properties(
            test_hybrid_metric_transport PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON)
    else()
        target_compile_features(
            test_hybrid_metric_transport PRIVATE cxx_std_20)
        set_target_properties(
            test_hybrid_metric_transport PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON)
    endif()
    warpx_set_compile_warnings(test_hybrid_metric_transport)

    add_test(
        NAME test_hybrid_metric_transport
        COMMAND test_hybrid_metric_transport)
endif()

if(WarpX_PRECISION STREQUAL "DOUBLE" AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    foreach(axis_correction IN ITEMS 0 1)
        add_warpx_test(
            test_rz_hybrid_axial_continuity_axis${axis_correction}
            RZ
            1
            "inputs_base_rz_hybrid_axial_continuity boundary.verboncoeur_axis_correction=${axis_correction}"
            "analysis_axial_continuity.py"
            OFF
            OFF
        )
    endforeach()
    add_warpx_test(
        test_rz_hybrid_axial_continuity_cubic_mpi
        RZ
        2
        "inputs_base_rz_hybrid_axial_continuity algo.particle_shape=3"
        "analysis_axial_continuity.py"
        OFF
        OFF
    )
    add_warpx_test(
        test_rz_hybrid_reflective_wall_continuity
        RZ
        2
        "inputs_base_rz_hybrid_axial_continuity max_step=4 warpx.const_dt=2.e-11 boundary.field_lo='none pec' boundary.field_hi='pec pec' boundary.particle_lo='none reflecting' boundary.particle_hi='reflecting reflecting' ions.ux=5.e-5 diag.intervals=4 diag.plot_raw_fields=1"
        "analysis_reflective_wall.py"
        OFF
        OFF
    )
endif()
