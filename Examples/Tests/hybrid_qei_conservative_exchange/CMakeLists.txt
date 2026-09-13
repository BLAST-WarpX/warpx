# Add tests (alphabetical order) ##############################################
#

if(RZ IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL DOUBLE
   AND WarpX_PARTICLE_PRECISION STREQUAL DOUBLE)
    add_executable(test_rz_pressure_work test_rz_pressure_work.cpp)
    target_link_libraries(test_rz_pressure_work PRIVATE lib_rz)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_rz_pressure_work)
    endif()
    warpx_set_compile_warnings(test_rz_pressure_work)
    foreach(case operators native)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pressure_work_${case})
        set(pressure_work_args)
        if(case STREQUAL operators)
            set(pressure_work_args test.operator_only=1)
        endif()
        add_test(NAME test_rz_pressure_work_${case}
            COMMAND test_rz_pressure_work ${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_rz_pressure_work
                ${pressure_work_args})
        set_tests_properties(test_rz_pressure_work_${case} PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=1"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pressure_work_${case})
    endforeach()
    foreach(case shape1 shape2 shape4 axial_walls quarter_axis compressing restart)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pressure_work_${case})
        if(case MATCHES "shape([124])")
            set(pressure_work_args "algo.particle_shape=${CMAKE_MATCH_1}")
        elseif(case STREQUAL axial_walls)
            set(pressure_work_args "boundary.field_lo=none pec" "boundary.field_hi=pec pec"
                "boundary.particle_lo=none reflecting" "boundary.particle_hi=reflecting reflecting")
        elseif(case STREQUAL quarter_axis)
            set(pressure_work_args boundary.verboncoeur_axis_correction=0)
        elseif(case STREQUAL compressing)
            set(pressure_work_args warpx.const_dt=2.e-12 test.require_electron_heating=1
                "hybrid_pic_model.initial_elec_pressure(x,y,z)=1.e6*(1+0.3*cos(pi*x/length))"
                ions.momentum_distribution_type=parse_momentum_function
                "ions.momentum_function_ux(x,y,z)=-2.e5*x/length/clight"
                "ions.momentum_function_uy(x,y,z)=-2.e5*y/length/clight"
                "ions.momentum_function_uz(x,y,z)=0")
        else()
            set(pressure_work_args
                "amr.restart=${CMAKE_CURRENT_BINARY_DIR}/pressure_work_native/diags/chk000200")
        endif()
        add_test(NAME test_rz_pressure_work_${case}
            COMMAND test_rz_pressure_work ${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_rz_pressure_work
                ${pressure_work_args})
        set_tests_properties(test_rz_pressure_work_${case} PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=1"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pressure_work_${case})
        if(case STREQUAL restart)
            set_tests_properties(test_rz_pressure_work_${case} PROPERTIES
                FIXTURES_REQUIRED rz_pressure_work_checkpoint)
        endif()
    endforeach()
    set_tests_properties(test_rz_pressure_work_native PROPERTIES
        FIXTURES_SETUP rz_pressure_work_checkpoint)
    set_tests_properties(test_rz_pressure_work_restart PROPERTIES
        FIXTURES_SETUP rz_pressure_work_restarted)
    add_test(NAME test_rz_pressure_work_restart_comparison
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_pressure_work_restart.py
            ${CMAKE_CURRENT_BINARY_DIR}/pressure_work_native
            ${CMAKE_CURRENT_BINARY_DIR}/pressure_work_restart)
    set_tests_properties(test_rz_pressure_work_restart_comparison PROPERTIES
        FIXTURES_REQUIRED "rz_pressure_work_checkpoint;rz_pressure_work_restarted"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pressure_work_restart)
    foreach(rejection changed_axis disabled magnetic)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pressure_work_reject_${rejection})
        set(pressure_reject_args
            "amr.restart=${CMAKE_CURRENT_BINARY_DIR}/pressure_work_native/diags/chk000200")
        if(rejection STREQUAL changed_axis)
            list(APPEND pressure_reject_args boundary.verboncoeur_axis_correction=0)
            set(pressure_expected "Invalid or changed RZ pressure-work checkpoint contract")
        elseif(rejection STREQUAL disabled)
            list(APPEND pressure_reject_args hybrid_pic_model.conservative_pressure_work=0
                hybrid_pic_model.conservative_pressure_work_pec=0)
            set(pressure_expected "Restart must preserve the RZ pressure-work model")
        else()
            set(pressure_reject_args test.inject_magnetic_current=1)
            set(pressure_expected "RZ conservative pressure work currently requires zero Ampere")
        endif()
        add_test(NAME test_rz_pressure_work_rejects_${rejection}
            COMMAND ${Python_EXECUTABLE}
                ${WarpX_SOURCE_DIR}/Examples/Tests/implicit/check_command_output.py
                --expect-failure --expect "${pressure_expected}" -- $<TARGET_FILE:test_rz_pressure_work>
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_rz_pressure_work ${pressure_reject_args})
        set_tests_properties(test_rz_pressure_work_rejects_${rejection} PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=1" FIXTURES_REQUIRED rz_pressure_work_checkpoint
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pressure_work_reject_${rejection})
    endforeach()
    if(WarpX_MPI)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pressure_work_mpi_restart)
        add_test(NAME test_rz_pressure_work_mpi_restart
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_rz_pressure_work> ${MPIEXEC_POSTFLAGS}
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_rz_pressure_work
                "amr.restart=${CMAKE_CURRENT_BINARY_DIR}/pressure_work_native/diags/chk000200")
        set_tests_properties(test_rz_pressure_work_mpi_restart PROPERTIES
            FIXTURES_REQUIRED rz_pressure_work_checkpoint PROCESSORS 2
            FIXTURES_SETUP rz_pressure_work_mpi_restarted
            ENVIRONMENT "OMP_NUM_THREADS=1"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pressure_work_mpi_restart)
        add_test(NAME test_rz_pressure_work_mpi_restart_comparison
            COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_pressure_work_restart.py
                ${CMAKE_CURRENT_BINARY_DIR}/pressure_work_native
                ${CMAKE_CURRENT_BINARY_DIR}/pressure_work_mpi_restart)
        set_tests_properties(test_rz_pressure_work_mpi_restart_comparison PROPERTIES
            FIXTURES_REQUIRED "rz_pressure_work_checkpoint;rz_pressure_work_mpi_restarted"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pressure_work_mpi_restart)
    endif()
    set(implicit_restart_procs 1)
    if(WarpX_MPI)
        set(implicit_restart_procs 2)
    endif()
    add_warpx_test(
        test_rz_pressure_initialized_qei
        RZ
        1
        inputs_base_rz_pressure_initialization
        "analysis_resolved_restart.py --implicit --initial-pressure 1.e6"
        OFF
        OFF)
    add_warpx_test(
        test_rz_pressure_initialized_qei_restart
        RZ
        ${implicit_restart_procs}
        "inputs_base_rz_pressure_initialization amr.restart='../test_rz_pressure_initialized_qei/diags/chk000100' hybrid_pic_model.initial_elec_pressure(x,y,z)=2.e6"
        "analysis_resolved_restart.py --implicit --initial-pressure 1.e6 --reference ../test_rz_pressure_initialized_qei"
        OFF
        test_rz_pressure_initialized_qei)
    if(TEST test_rz_pressure_initialized_qei.run)
        foreach(rejection vacuum conflict)
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pressure_${rejection})
            if(rejection STREQUAL vacuum)
                set(pressure_override "ions.density_function(x,y,z)=0")
                set(pressure_expected "Initial electron pressure is invalid")
            else()
                set(pressure_override "hybrid_pic_model.initial_elec_temp(x,y,z)=100")
                set(pressure_expected "Choose either initial_elec_pressure")
            endif()
            add_test(NAME test_rz_pressure_initialization_rejects_${rejection}
                COMMAND ${Python_EXECUTABLE}
                ${WarpX_SOURCE_DIR}/Examples/Tests/implicit/check_command_output.py
                --expect-failure --expect "${pressure_expected}" -- $<TARGET_FILE:app_rz>
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_rz_pressure_initialization
                "${pressure_override}")
            set_tests_properties(test_rz_pressure_initialization_rejects_${rejection}
                PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1"
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pressure_${rejection})
        endforeach()
    endif()
    add_warpx_test(
        test_rz_implicit_qei_moving
        RZ
        1
        "inputs_base_rz_resolved_restart hybrid_pic_model.electron_energy_transport=finite_volume_implicit"
        "analysis_resolved_restart.py --implicit"
        OFF
        OFF)
    add_warpx_test(
        test_rz_implicit_qei_moving_restart
        RZ
        ${implicit_restart_procs}
        "inputs_base_rz_resolved_restart hybrid_pic_model.electron_energy_transport=finite_volume_implicit amr.restart='../test_rz_implicit_qei_moving/diags/chk000100'"
        "analysis_resolved_restart.py --implicit --reference ../test_rz_implicit_qei_moving"
        OFF
        test_rz_implicit_qei_moving)
    add_warpx_test(
        test_rz_resolved_qei_moving
        RZ
        1
        inputs_base_rz_resolved_restart
        analysis_resolved_restart.py
        OFF
        OFF)
    add_warpx_test(
        test_rz_resolved_qei_moving_restart
        RZ
        1
        "inputs_base_rz_resolved_restart amr.restart='../test_rz_resolved_qei_moving/diags/chk000100'"
        "analysis_resolved_restart.py --reference ../test_rz_resolved_qei_moving"
        OFF
        test_rz_resolved_qei_moving)
    if(WarpX_MPI)
        add_warpx_test(
            test_rz_resolved_qei_moving_mpi_restart
            RZ
            2
            "inputs_base_rz_resolved_restart amr.restart='../test_rz_resolved_qei_moving/diags/chk000100'"
            "analysis_resolved_restart.py --reference ../test_rz_resolved_qei_moving"
            OFF
            test_rz_resolved_qei_moving)
    endif()
    if(TEST test_rz_resolved_qei_moving.run)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/resolved_schema)
        add_test(NAME test_rz_resolved_qei_schema COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_resolved_schema.py
            $<TARGET_FILE:app_rz> ${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_rz_resolved_restart
            ${CMAKE_BINARY_DIR}/bin/test_rz_resolved_qei_moving/diags/chk000100)
        set_tests_properties(test_rz_resolved_qei_schema PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/resolved_schema
            FIXTURES_REQUIRED resolved_qei_checkpoint)
        set_property(TEST test_rz_resolved_qei_moving.run
            APPEND PROPERTY FIXTURES_SETUP resolved_qei_checkpoint)
        if(TEST test_rz_resolved_qei_moving.cleanup)
            set_property(TEST test_rz_resolved_qei_moving.cleanup
                APPEND PROPERTY FIXTURES_CLEANUP resolved_qei_checkpoint)
        endif()
    endif()
    add_executable(test_rz_resolved_qei_support test_resolved_support.cpp)
    target_link_libraries(test_rz_resolved_qei_support PRIVATE lib_rz)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_rz_resolved_qei_support)
    endif()
    warpx_set_compile_warnings(test_rz_resolved_qei_support)
    add_test(NAME test_rz_resolved_qei_support
        COMMAND test_rz_resolved_qei_support ${CMAKE_CURRENT_SOURCE_DIR}/inputs_resolved_support
            amrex.the_arena_init_size=0)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/resolved_serial
                        ${CMAKE_CURRENT_BINARY_DIR}/resolved_source
                        ${CMAKE_CURRENT_BINARY_DIR}/resolved_zero
                        ${CMAKE_CURRENT_BINARY_DIR}/resolved_mpi)
    set_tests_properties(test_rz_resolved_qei_support PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/resolved_serial)
    add_test(NAME test_rz_native_caloric_source
        COMMAND test_rz_resolved_qei_support ${CMAKE_CURRENT_SOURCE_DIR}/inputs_resolved_support
            test.check_radiation_source=1 amrex.the_arena_init_size=0)
    set_tests_properties(test_rz_native_caloric_source PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/resolved_source)
    add_test(NAME test_rz_resolved_qei_zero_rate
        COMMAND test_rz_resolved_qei_support ${CMAKE_CURRENT_SOURCE_DIR}/inputs_resolved_support
            "hybrid_pic_model.electron_ion_relaxation_rate(rho,Te,Ti,t)=0"
            test.expect_no_exchange=1 amrex.the_arena_init_size=0)
    set_tests_properties(test_rz_resolved_qei_zero_rate PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/resolved_zero)
    if(WarpX_MPI)
        add_test(NAME test_rz_resolved_qei_support_mpi
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPI_ALLOW_ROOT} ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_POSTFLAGS} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_rz_resolved_qei_support>
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_resolved_support
                test.check_radiation_source=1 amrex.the_arena_init_size=0)
        set_tests_properties(test_rz_resolved_qei_support_mpi PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=1" PROCESSORS 2
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/resolved_mpi)
    endif()
endif()

list(GET WarpX_DIMS 0 source_noop_dim)
warpx_set_suffix_dims(source_noop_suffix ${source_noop_dim})
add_executable(test_electron_caloric_noop test_electron_caloric_noop.cpp)
target_link_libraries(test_electron_caloric_noop PRIVATE lib_${source_noop_suffix})
if(WarpX_COMPUTE STREQUAL CUDA)
    setup_target_for_cuda_compilation(test_electron_caloric_noop)
endif()
warpx_set_compile_warnings(test_electron_caloric_noop)
add_test(NAME test_electron_caloric_noop COMMAND test_electron_caloric_noop
    amrex.the_arena_init_size=0)
set_tests_properties(test_electron_caloric_noop PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")

add_warpx_test(
    test_2d_hybrid_qei_conservative_exchange  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_hybrid_qei_conservative_exchange  # inputs
    "analysis.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (analytic exchange and conservative ledgers are checked)
    OFF  # dependency
)
