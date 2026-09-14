# Add tests (alphabetical order) ##############################################
#

find_package(Python REQUIRED COMPONENTS Interpreter)

if(WarpX_LIB AND RZ IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL DOUBLE)
    add_executable(test_rz_native_conduction test_rz_native_conduction.cpp)
    target_link_libraries(test_rz_native_conduction PRIVATE lib_rz)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_rz_native_conduction)
        target_compile_features(test_rz_native_conduction PRIVATE cuda_std_20)
    else()
        target_compile_features(test_rz_native_conduction PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_rz_native_conduction)
    set(rz_native_cases serial mpi)
    if(WarpX_SINGULARITY_EOS)
        list(APPEND rz_native_cases table_serial table_mpi)
        set(rz_table_fixture_dir "${CMAKE_CURRENT_BINARY_DIR}/nonlinear_conduction_fixture")
        set(rz_table_fixture "${rz_table_fixture_dir}/electron_only_fixture.sp5")
        if(NOT 1 IN_LIST WarpX_DIMS)
            add_test(NAME nonlinear_conduction_eos_fixture
                COMMAND ${CMAKE_COMMAND} -E env
                    "WARPX_SPINER_FIXTURE_PATH=${rz_table_fixture}"
                    "WARPX_SPINER_FIXTURE_NUM_T=512"
                    $<TARGET_FILE:test_hybrid_electron_thermodynamics_spiner>)
            set_tests_properties(nonlinear_conduction_eos_fixture PROPERTIES
                FIXTURES_SETUP nonlinear_conduction_eos_fixture ENVIRONMENT "OMP_NUM_THREADS=2")
        endif()
    endif()
    foreach(native_case IN LISTS rz_native_cases)
        if(native_case MATCHES "mpi$" AND NOT WarpX_MPI)
            continue()
        endif()
        set(native_dir "${CMAKE_CURRENT_BINARY_DIR}/rz_native_conduction_${native_case}")
        file(MAKE_DIRECTORY ${native_dir})
        set(native_launcher)
        set(native_postflags)
        set(native_input inputs_rz_native_nonlinear_conduction)
        set(native_arguments)
        if(native_case MATCHES "^table")
            set(native_input inputs_rz_native_table_conduction)
            list(APPEND native_arguments "hybrid_pic_model.electron_eos_ions_table_file=${rz_table_fixture}")
        endif()
        if(native_case MATCHES "mpi$")
            set(native_launcher ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS})
            set(native_postflags ${MPIEXEC_POSTFLAGS})
        endif()
        add_test(NAME test_rz_native_conduction_${native_case}.run COMMAND ${native_launcher}
            $<TARGET_FILE:test_rz_native_conduction> ${native_postflags} ${native_input} ${native_arguments}
            amrex.abort_on_unused_inputs=1)
        set_tests_properties(test_rz_native_conduction_${native_case}.run PROPERTIES
            WORKING_DIRECTORY ${native_dir} FIXTURES_SETUP rz_native_conduction_${native_case}
            ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
        if(native_case MATCHES "^table")
            set_tests_properties(test_rz_native_conduction_${native_case}.run PROPERTIES
                FIXTURES_REQUIRED nonlinear_conduction_eos_fixture)
        endif()
        if(native_case MATCHES "mpi$")
            set_tests_properties(test_rz_native_conduction_${native_case}.run PROPERTIES PROCESSORS 2 RUN_SERIAL TRUE)
        endif()
        add_test(NAME test_rz_native_conduction_${native_case}.analysis COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_rz_native_conduction.py ${native_dir})
        set_tests_properties(test_rz_native_conduction_${native_case}.analysis PROPERTIES
            FIXTURES_REQUIRED rz_native_conduction_${native_case})
    endforeach()
    add_executable(test_rz_nonlinear_electron_conduction test_nonlinear_electron_conduction.cpp)
    target_link_libraries(test_rz_nonlinear_electron_conduction PRIVATE lib_rz)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_rz_nonlinear_electron_conduction)
        target_compile_features(test_rz_nonlinear_electron_conduction PRIVATE cuda_std_20)
    else()
        target_compile_features(test_rz_nonlinear_electron_conduction PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_rz_nonlinear_electron_conduction)
    set(rz_nonlinear_dir "${CMAKE_CURRENT_BINARY_DIR}/rz_nonlinear_conduction")
    file(MAKE_DIRECTORY ${rz_nonlinear_dir})
    add_test(NAME test_rz_nonlinear_conduction_latent COMMAND ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/analysis_nonlinear_electron_conduction.py
        $<TARGET_FILE:test_rz_nonlinear_electron_conduction> --rz)
    set_tests_properties(test_rz_nonlinear_conduction_latent PROPERTIES
        WORKING_DIRECTORY ${rz_nonlinear_dir} ENVIRONMENT "OMP_NUM_THREADS=2")
    add_test(NAME test_rz_nonlinear_conduction_budget COMMAND test_rz_nonlinear_electron_conduction
        test.reject_budget=1 amrex.throw_exception=1 amrex.the_arena_init_size=0)
    set_tests_properties(test_rz_nonlinear_conduction_budget PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")
    if(WarpX_SINGULARITY_EOS)
        foreach(table_case IN ITEMS table table_negative)
            set(table_path "${rz_table_fixture}")
            if(table_case STREQUAL table_negative)
                set(table_path "${rz_table_fixture_dir}/electron_only_fixture_negative_reference.sp5")
            endif()
            set(table_dir "${CMAKE_CURRENT_BINARY_DIR}/rz_nonlinear_conduction_${table_case}")
            file(MAKE_DIRECTORY ${table_dir})
            add_test(NAME test_rz_nonlinear_conduction_${table_case} COMMAND ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/analysis_nonlinear_electron_conduction.py
                $<TARGET_FILE:test_rz_nonlinear_electron_conduction> --rz --table-file ${table_path})
            set_tests_properties(test_rz_nonlinear_conduction_${table_case} PROPERTIES
                WORKING_DIRECTORY ${table_dir} ENVIRONMENT "OMP_NUM_THREADS=2"
                FIXTURES_REQUIRED nonlinear_conduction_eos_fixture)
        endforeach()
    endif()
    if(WarpX_MPI)
        set(rz_nonlinear_mpi_dir "${CMAKE_CURRENT_BINARY_DIR}/rz_nonlinear_conduction_mpi")
        file(MAKE_DIRECTORY ${rz_nonlinear_mpi_dir})
        add_test(NAME test_rz_nonlinear_conduction_latent_mpi COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_nonlinear_electron_conduction.py
            $<TARGET_FILE:test_rz_nonlinear_electron_conduction> --rz --mpi-exec ${MPIEXEC_EXECUTABLE})
        set_tests_properties(test_rz_nonlinear_conduction_latent_mpi PROPERTIES
            PROCESSORS 2 RUN_SERIAL TRUE WORKING_DIRECTORY ${rz_nonlinear_mpi_dir}
            ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
endif()

if(WarpX_LIB AND 1 IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL DOUBLE)
    add_executable(test_nonlinear_electron_conduction test_nonlinear_electron_conduction.cpp)
    target_link_libraries(test_nonlinear_electron_conduction PRIVATE lib_1d)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_nonlinear_electron_conduction)
        target_compile_features(test_nonlinear_electron_conduction PRIVATE cuda_std_20)
    else()
        target_compile_features(test_nonlinear_electron_conduction PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_nonlinear_electron_conduction)
    add_test(NAME test_1d_nonlinear_conduction_budget
        COMMAND test_nonlinear_electron_conduction test.reject_budget=1
            amrex.throw_exception=1 amrex.the_arena_init_size=0)
    set_tests_properties(test_1d_nonlinear_conduction_budget PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")
    set(nonlinear_cases latent)
    if(WarpX_SINGULARITY_EOS)
        list(APPEND nonlinear_cases table table_negative)
        set(nonlinear_fixture_dir "${CMAKE_CURRENT_BINARY_DIR}/nonlinear_conduction_fixture")
        set(nonlinear_fixture "${nonlinear_fixture_dir}/electron_only_fixture.sp5")
        add_test(NAME nonlinear_conduction_eos_fixture
            COMMAND ${CMAKE_COMMAND} -E env
                "WARPX_SPINER_FIXTURE_PATH=${nonlinear_fixture}"
                "WARPX_SPINER_FIXTURE_NUM_T=512"
                $<TARGET_FILE:test_hybrid_electron_thermodynamics_spiner>)
        set_tests_properties(nonlinear_conduction_eos_fixture PROPERTIES
            FIXTURES_SETUP nonlinear_conduction_eos_fixture ENVIRONMENT "OMP_NUM_THREADS=2")
    endif()
    foreach(nonlinear_case IN LISTS nonlinear_cases)
        set(nonlinear_name test_1d_nonlinear_conduction_${nonlinear_case})
        set(nonlinear_args)
        if(nonlinear_case STREQUAL table)
            list(APPEND nonlinear_args --table-file "${nonlinear_fixture}")
        elseif(nonlinear_case STREQUAL table_negative)
            list(APPEND nonlinear_args --table-file "${nonlinear_fixture_dir}/electron_only_fixture_negative_reference.sp5")
        endif()
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${nonlinear_name}")
        add_test(NAME ${nonlinear_name}
            COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_nonlinear_electron_conduction.py
                $<TARGET_FILE:test_nonlinear_electron_conduction> ${nonlinear_args})
        set_tests_properties(${nonlinear_name} PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${nonlinear_name}"
            ENVIRONMENT "OMP_NUM_THREADS=2")
        if(nonlinear_case MATCHES "^table")
            set_tests_properties(${nonlinear_name} PROPERTIES FIXTURES_REQUIRED nonlinear_conduction_eos_fixture)
        endif()
    endforeach()
    if(WarpX_MPI)
        set(nonlinear_mpi_dir "${CMAKE_CURRENT_BINARY_DIR}/nonlinear_conduction_latent_mpi")
        file(MAKE_DIRECTORY ${nonlinear_mpi_dir})
        add_test(NAME test_1d_nonlinear_conduction_latent_mpi
            COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_nonlinear_electron_conduction.py
                $<TARGET_FILE:test_nonlinear_electron_conduction> --mpi-exec "${MPIEXEC_EXECUTABLE}")
        set_tests_properties(test_1d_nonlinear_conduction_latent_mpi PROPERTIES
            PROCESSORS 2 WORKING_DIRECTORY ${nonlinear_mpi_dir} ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
    if(WarpX_SINGULARITY_EOS AND WarpX_APP)
        add_warpx_test(
            test_1d_nonlinear_conduction_native_table
            1
            1
            "inputs_test_1d_nonlinear_conduction_native_table hybrid_pic_model.electron_eos_ions_table_file='${nonlinear_fixture}'"
            "analysis_nonlinear_conduction_native_table.py diags/diag000000 diags/diag000064"
            OFF OFF)
        set_tests_properties(test_1d_nonlinear_conduction_native_table.run PROPERTIES
            FIXTURES_REQUIRED nonlinear_conduction_eos_fixture)
    endif()
endif()

if(WarpX_PYTHON)
    add_test(NAME test_species_qei_rates_picmi
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_species_rates_picmi.py)
    set_tests_properties(test_species_qei_rates_picmi PROPERTIES
        ENVIRONMENT "PYTHONPATH=${WarpX_SOURCE_DIR}/Python")
endif()

if(WarpX_LIB AND WarpX_PRECISION STREQUAL DOUBLE)
    foreach(dim IN LISTS WarpX_DIMS)
        if(dim STREQUAL RCYLINDER OR dim STREQUAL RSPHERE)
            continue()
        endif()
        warpx_set_suffix_dims(suffix ${dim})
        set(conduction_target test_${suffix}_electron_heat_conduction)
        add_executable(${conduction_target} test_electron_heat_conduction.cpp)
        target_link_libraries(${conduction_target} PRIVATE lib_${suffix})
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(${conduction_target})
        endif()
        warpx_set_compile_warnings(${conduction_target})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_${suffix})
        add_test(NAME ${conduction_target} COMMAND ${conduction_target})
        set_tests_properties(${conduction_target} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_${suffix})
        add_test(NAME ${conduction_target}_invalid
            COMMAND ${Python_EXECUTABLE}
                ${WarpX_SOURCE_DIR}/Examples/Tests/implicit/check_command_output.py
                --expect-failure --expect "conductivity and temperature must be finite" --
                $<TARGET_FILE:${conduction_target}> test.invalid_conductivity=1)
        set_tests_properties(${conduction_target}_invalid PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_${suffix})
        if(dim STREQUAL RZ)
            add_test(NAME ${conduction_target}_invalid_composition
                COMMAND ${Python_EXECUTABLE}
                    ${WarpX_SOURCE_DIR}/Examples/Tests/implicit/check_command_output.py
                    --expect-failure --expect "Electron charge moments require consistent" --
                    $<TARGET_FILE:${conduction_target}> test.invalid_composition=1)
            set_tests_properties(${conduction_target}_invalid_composition PROPERTIES
                ENVIRONMENT "OMP_NUM_THREADS=1"
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_${suffix})
        endif()
        if(WarpX_MPI)
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_${suffix}_mpi)
            add_test(NAME ${conduction_target}_mpi
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:${conduction_target}> ${MPIEXEC_POSTFLAGS})
            set_tests_properties(${conduction_target}_mpi PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1"
                PROCESSORS 2 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_${suffix}_mpi)
        endif()
    endforeach()
endif()

if(RZ IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL DOUBLE
   AND WarpX_PARTICLE_PRECISION STREQUAL DOUBLE)
    add_executable(test_rz_species_qei_rates test_species_rates.cpp)
    target_link_libraries(test_rz_species_qei_rates PRIVATE lib_rz)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_rz_species_qei_rates)
    endif()
    warpx_set_compile_warnings(test_rz_species_qei_rates)
    foreach(case specific fallback legacy legacy_fallback)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/species_rates_${case})
        set(rate_args)
        if(case MATCHES fallback)
            list(APPEND rate_args test.fallback=1)
        endif()
        if(case MATCHES legacy)
            list(APPEND rate_args test.legacy=1)
        endif()
        add_test(NAME test_rz_species_qei_rates_${case}
            COMMAND test_rz_species_qei_rates ${CMAKE_CURRENT_SOURCE_DIR}/inputs_species_rates ${rate_args})
        set_tests_properties(test_rz_species_qei_rates_${case} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/species_rates_${case})
    endforeach()
    foreach(case specific legacy)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/species_rates_invalid_${case})
        set(invalid_rate_args test.invalid_rate=1)
        if(case STREQUAL legacy)
            list(APPEND invalid_rate_args test.legacy=1)
        endif()
        add_test(NAME test_rz_species_qei_rates_invalid_${case}
            COMMAND ${Python_EXECUTABLE}
                ${WarpX_SOURCE_DIR}/Examples/Tests/implicit/check_command_output.py
                --expect-failure --expect "Qei rate or caloric update is invalid" --
                $<TARGET_FILE:test_rz_species_qei_rates>
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_species_rates ${invalid_rate_args})
        set_tests_properties(test_rz_species_qei_rates_invalid_${case} PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=1"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/species_rates_invalid_${case})
    endforeach()
    if(WarpX_MPI)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/species_rates_mpi)
        add_test(NAME test_rz_species_qei_rates_mpi
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_rz_species_qei_rates>
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_species_rates ${MPIEXEC_POSTFLAGS})
        set_tests_properties(test_rz_species_qei_rates_mpi PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=1" PROCESSORS 2
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/species_rates_mpi)
    endif()
    foreach(case native mpi ambiguous changed_signature)
        if(case STREQUAL mpi AND NOT WarpX_MPI)
            continue()
        endif()
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/composition_conduction_${case})
        set(composition_launch)
        set(composition_args test.composition_conduction=1)
        if(case STREQUAL mpi)
            set(composition_launch ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS})
        elseif(case STREQUAL ambiguous)
            set(composition_launch ${Python_EXECUTABLE}
                ${WarpX_SOURCE_DIR}/Examples/Tests/implicit/check_command_output.py
                --expect-failure --expect "Specify exactly one electron conductivity signature" --)
            list(APPEND composition_args test.ambiguous_conductivity=1)
        elseif(case STREQUAL changed_signature)
            set(composition_launch ${Python_EXECUTABLE}
                ${WarpX_SOURCE_DIR}/Examples/Tests/implicit/check_command_output.py
                --expect-failure --expect "Invalid or changed electron heat-conduction" --)
            list(APPEND composition_args test.changed_conduction_signature=1)
        endif()
        add_test(NAME test_rz_composition_conduction_${case}
            COMMAND ${composition_launch} $<TARGET_FILE:test_rz_species_qei_rates>
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_species_rates ${composition_args})
        set_tests_properties(test_rz_composition_conduction_${case} PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=1"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/composition_conduction_${case})
        if(case STREQUAL mpi)
            set_property(TEST test_rz_composition_conduction_${case} PROPERTY PROCESSORS 2)
        endif()
    endforeach()
    set(conduction_native_args warpx.const_dt=2.e-12
        hybrid_pic_model.electron_heat_conduction=1
        "hybrid_pic_model.electron_thermal_conductivity(rho,Te)=100"
        hybrid_pic_model.electron_conduction_flux_limiter=0
        "diag.fields_to_plot=rho Te Pe hybrid_conduction_energy_fp")
    foreach(case native restart mpi_restart)
        if(case STREQUAL mpi_restart AND NOT WarpX_MPI)
            continue()
        endif()
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_${case})
        set(conduction_restart_args)
        set(conduction_launch)
        if(NOT case STREQUAL native)
            set(conduction_restart_args
                "amr.restart=${CMAKE_CURRENT_BINARY_DIR}/conduction_native/diags/chk000200")
        endif()
        if(case STREQUAL mpi_restart)
            set(conduction_launch ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS})
        endif()
        add_test(NAME test_rz_electron_conduction_${case}
            COMMAND ${conduction_launch} $<TARGET_FILE:test_rz_pressure_work>
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_rz_pressure_work
                ${conduction_native_args} ${conduction_restart_args})
        set_tests_properties(test_rz_electron_conduction_${case} PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=1"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_${case}
            FIXTURES_SETUP electron_conduction_${case})
        if(NOT case STREQUAL native)
            set_tests_properties(test_rz_electron_conduction_${case} PROPERTIES
                FIXTURES_REQUIRED electron_conduction_native)
            add_test(NAME test_rz_electron_conduction_${case}_comparison
                COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_pressure_work_restart.py
                    ${CMAKE_CURRENT_BINARY_DIR}/conduction_native
                    ${CMAKE_CURRENT_BINARY_DIR}/conduction_${case})
            set_tests_properties(test_rz_electron_conduction_${case}_comparison PROPERTIES
                FIXTURES_REQUIRED "electron_conduction_native;electron_conduction_${case}"
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_${case})
        endif()
        if(case STREQUAL mpi_restart)
            set_tests_properties(test_rz_electron_conduction_${case} PROPERTIES PROCESSORS 2)
        endif()
    endforeach()
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_rejection)
    add_test(NAME test_rz_electron_conduction_changed_restart
        COMMAND ${Python_EXECUTABLE} ${WarpX_SOURCE_DIR}/Examples/Tests/implicit/check_command_output.py
            --expect-failure --expect "Invalid or changed electron heat-conduction checkpoint" --
            $<TARGET_FILE:test_rz_pressure_work>
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_rz_pressure_work
            ${conduction_native_args}
            "amr.restart=${CMAKE_CURRENT_BINARY_DIR}/conduction_native/diags/chk000200"
            "hybrid_pic_model.electron_thermal_conductivity(rho,Te)=101")
    set_tests_properties(test_rz_electron_conduction_changed_restart PROPERTIES
        FIXTURES_REQUIRED electron_conduction_native ENVIRONMENT "OMP_NUM_THREADS=1"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_rejection)
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
