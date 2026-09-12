# Add tests (alphabetical order) ##############################################
#

if(WarpX_PRECISION STREQUAL "DOUBLE")
    set(coupled_runtime_procs 1)
    if(WarpX_MPI)
        set(coupled_runtime_procs 2)
    endif()
    if(WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
        # Native material-wall prerequisite; radiation is deliberately disabled.
        add_warpx_test(
            test_1d_radiation_transport_material_wall_adjoint
            1
            1
            "inputs_base_1d_material_wall_control hybrid_pic_model.conservative_pressure_work_pec=1"
            "analysis_material_wall_control.py ."
            OFF
            OFF
        )
        add_warpx_test(
            test_1d_radiation_transport_material_wall_adjoint_restart
            1
            ${coupled_runtime_procs}
            "inputs_base_1d_material_wall_control hybrid_pic_model.conservative_pressure_work_pec=1 amr.restart='../test_1d_radiation_transport_material_wall_adjoint/diags/chk001600'"
            "analysis_material_wall_control.py . --reference ../test_1d_radiation_transport_material_wall_adjoint"
            OFF
            test_1d_radiation_transport_material_wall_adjoint
        )
        if(TEST test_1d_radiation_transport_material_wall_adjoint.run)
            set(history_schema_dir "${CMAKE_CURRENT_BINARY_DIR}/hybrid_history_schema")
            file(MAKE_DIRECTORY ${history_schema_dir})
            add_test(NAME test_1d_hybrid_moment_history_schema COMMAND
                ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_hybrid_history_schema.py
                $<TARGET_FILE:app_1d> ${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_1d_material_wall_control
                ${CMAKE_BINARY_DIR}/bin/test_1d_radiation_transport_material_wall_adjoint
                --probe-executable $<TARGET_FILE:test_1d_particle_impulse>)
            set_tests_properties(test_1d_hybrid_moment_history_schema PROPERTIES
                WORKING_DIRECTORY ${history_schema_dir}
                FIXTURES_REQUIRED native_pec_wall_checkpoint)
            set_property(TEST test_1d_radiation_transport_material_wall_adjoint.run
                APPEND PROPERTY FIXTURES_SETUP native_pec_wall_checkpoint)
            set_property(TEST test_1d_radiation_transport_material_wall_adjoint_restart.run
                test_1d_radiation_transport_material_wall_adjoint_restart.analysis
                APPEND PROPERTY FIXTURES_REQUIRED native_pec_wall_checkpoint)
            if(TEST test_1d_radiation_transport_material_wall_adjoint.cleanup)
                set_property(TEST test_1d_radiation_transport_material_wall_adjoint.cleanup
                    APPEND PROPERTY FIXTURES_CLEANUP native_pec_wall_checkpoint)
            endif()
        endif()
        add_warpx_test(
            test_1d_radiation_transport_moving_beam_ode
            1
            ${coupled_runtime_procs}
            "inputs_base_1d_moving_moment_beam"
            "analysis_moving_moment_beam.py . --check-momentum-diagnostic"
            OFF
            OFF
        )
        add_warpx_test(
            test_1d_radiation_transport_moving_beam_restart
            1
            1
            "inputs_base_1d_moving_moment_beam amr.restart='../test_1d_radiation_transport_moving_beam_ode/diags/chk000200'"
            "analysis_moving_moment_beam.py . --check-momentum-diagnostic --restart-reference ../test_1d_radiation_transport_moving_beam_ode"
            OFF
            test_1d_radiation_transport_moving_beam_ode
        )
        add_warpx_test(
            test_1d_radiation_transport_moving_moment_pulse
            1
            ${coupled_runtime_procs}
            "inputs_base_1d_moving_moment_pulse max_step=400 amr.n_cell=128 amr.max_grid_size=64 warpx.const_dt=8.333333333333333e-12 chk.intervals=200"
            "analysis_moving_moment_pulse.py ."
            OFF
            OFF
        )
        add_warpx_test(
            test_1d_radiation_transport_moving_moment_restart
            1
            1
            "inputs_base_1d_moving_moment_pulse max_step=400 amr.n_cell=128 amr.max_grid_size=64 warpx.const_dt=8.333333333333333e-12 chk.intervals=200 amr.restart='../test_1d_radiation_transport_moving_moment_pulse/diags/chk000200'"
            "analysis_moving_moment_pulse.py . --reference ../test_1d_radiation_transport_moving_moment_pulse"
            OFF
            test_1d_radiation_transport_moving_moment_pulse
        )
        # Wiring/transaction gates, not substitutes for the full oblique study.
        add_warpx_test(
            test_2d_radiation_transport_moving_moment_adapter
            2
            ${coupled_runtime_procs}
            "inputs_base_2d_moving_moment_oblique amr.n_cell=16 32 amr.max_grid_size=16 max_step=16 diag1.intervals=8 chk.intervals=8"
            "analysis_moving_moment_oblique.py . --probe"
            OFF
            OFF
        )
        add_warpx_test(
            test_2d_radiation_transport_moving_moment_adapter_restart
            2
            1
            "inputs_base_2d_moving_moment_oblique amr.n_cell=16 32 amr.max_grid_size=16 max_step=16 diag1.intervals=8 chk.intervals=8 amr.restart='../test_2d_radiation_transport_moving_moment_adapter/diags/chk000008'"
            "analysis_moving_moment_oblique.py . --probe --reference ../test_2d_radiation_transport_moving_moment_adapter"
            OFF
            test_2d_radiation_transport_moving_moment_adapter
        )
    endif()
    add_warpx_test(
        test_3d_radiation_transport_coupled_implicit
        3
        1
        inputs_base_3d_radiation_transport_coupled_implicit
        "analysis_coupled_implicit.py"
        OFF
        OFF
    )
    add_warpx_test(
        test_1d_radiation_transport_coupled_implicit
        1
        1
        inputs_base_1d_radiation_transport_coupled_implicit
        "analysis_coupled_implicit.py"
        OFF
        OFF
    )
    add_warpx_test(
        test_1d_radiation_transport_coupled_implicit_restart
        1
        2
        "inputs_base_1d_radiation_transport_coupled_implicit amr.restart='../test_1d_radiation_transport_coupled_implicit/diags/chk000010'"
        "analysis_coupled_implicit.py --reference ../test_1d_radiation_transport_coupled_implicit"
        OFF
        test_1d_radiation_transport_coupled_implicit
    )
    add_warpx_test(
        test_1d_radiation_transport_coupled_retry_details
        1
        ${coupled_runtime_procs}
        "inputs_base_1d_radiation_transport_coupled_implicit radiation_transport.coupled_max_subdivisions=6 radiation_transport.lte_exchange_max_iterations=4 radiation_energy.include_solver_details=1"
        "analysis_coupled_implicit.py --solver-details"
        OFF
        OFF
    )
    add_warpx_test(
        test_1d_radiation_transport_coupled_retry_details_restart
        1
        1
        "inputs_base_1d_radiation_transport_coupled_implicit radiation_transport.coupled_max_subdivisions=6 radiation_transport.lte_exchange_max_iterations=4 radiation_energy.include_solver_details=1 amr.restart='../test_1d_radiation_transport_coupled_retry_details/diags/chk000010'"
        "analysis_coupled_implicit.py --solver-details --reference ../test_1d_radiation_transport_coupled_retry_details"
        OFF
        test_1d_radiation_transport_coupled_retry_details
    )
endif()

add_warpx_test(
    test_1d_radiation_transport_initial_blackbody
    1
    1
    inputs_base_1d_radiation_transport_initial_blackbody
    "analysis_initial_blackbody.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)

if(WarpX_PRECISION STREQUAL DOUBLE)
    add_warpx_test(
        test_1d_radiation_transport_implicit_diffusion
        1
        2
        inputs_base_1d_radiation_transport_implicit_diffusion
        analysis_implicit_diffusion.py
        OFF
        OFF
    )
    add_warpx_test(
        test_3d_radiation_transport_implicit_diffusion
        3
        2
        inputs_base_3d_radiation_transport_implicit_diffusion
        analysis_implicit_diffusion.py
        OFF
        OFF
    )
endif()

add_warpx_test(
    test_1d_radiation_transport_bath
    1
    1
    inputs_base_1d_radiation_transport_bath
    "analysis_bath.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)
add_warpx_test(
    test_1d_radiation_transport_bath_restart
    1
    2
    "inputs_base_1d_radiation_transport_bath amr.restart='../test_1d_radiation_transport_bath/diags/chk000050'"
    "analysis_bath.py --reference ../test_1d_radiation_transport_bath --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    test_1d_radiation_transport_bath
)

add_warpx_test(
    test_1d_radiation_transport_bath_temperature
    1
    1
    inputs_base_1d_radiation_transport_bath_temperature
    "analysis_bath.py --temperature --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)
add_warpx_test(
    test_3d_radiation_transport_bath
    3
    1
    inputs_base_3d_radiation_transport_bath
    "analysis_bath.py --dims 3 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)

add_warpx_test(
    test_3d_radiation_transport_group_sampling
    3
    1
    inputs_base_3d_radiation_transport_group_sampling
    "analysis_group_sampling.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)
add_warpx_test(
    test_3d_radiation_transport_packet_batches
    3
    1
    "inputs_base_3d_radiation_transport_group_sampling radiation_transport.particle_conversion_batch_size=3"
    "analysis_group_sampling.py --reference ../test_3d_radiation_transport_group_sampling --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    test_3d_radiation_transport_group_sampling
)
add_warpx_test(
    test_3d_radiation_transport_packet_batches_mpi
    3
    2
    "inputs_base_3d_radiation_transport_group_sampling radiation_transport.particle_conversion_batch_size=3 amr.max_grid_size=2"
    "analysis_group_sampling.py --reference ../test_3d_radiation_transport_group_sampling --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    test_3d_radiation_transport_group_sampling
)
add_warpx_test(
    test_3d_radiation_transport_packet_batches_empty_rank
    3
    2
    "inputs_base_3d_radiation_transport_group_sampling radiation_transport.particle_conversion_batch_size=3"
    "analysis_group_sampling.py --reference ../test_3d_radiation_transport_group_sampling --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    test_3d_radiation_transport_group_sampling
)
add_warpx_test(
    test_3d_radiation_transport_packet_batches_checkpoint
    3
    1
    "inputs_base_3d_radiation_transport_conversion_random_seed_restart radiation_transport.particle_conversion_batch_size=3 radiation_transport.particle_conversion_packets_per_cell=8"
    "analysis_conversion_random_seed_restart.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)
add_warpx_test(
    test_3d_radiation_transport_packet_batches_restart
    3
    2
    "inputs_base_3d_radiation_transport_conversion_random_seed_restart_restart radiation_transport.particle_conversion_batch_size=3 radiation_transport.particle_conversion_packets_per_cell=8 amr.restart='../test_3d_radiation_transport_packet_batches_checkpoint/diags/chk000001'"
    "analysis_conversion_random_seed_restart.py --reference ../test_3d_radiation_transport_packet_batches_checkpoint --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    test_3d_radiation_transport_packet_batches_checkpoint
)
add_warpx_test(
    test_3d_radiation_transport_group_sampling_streaming
    3
    1
    inputs_test_3d_radiation_transport_group_sampling_streaming
    "analysis_group_sampling.py --streaming --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)
add_warpx_test(
    test_3d_radiation_transport_group_sampling_mpi
    3
    2
    "inputs_base_3d_radiation_transport_group_sampling amr.max_grid_size=2"
    "analysis_group_sampling.py --reference ../test_3d_radiation_transport_group_sampling --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    test_3d_radiation_transport_group_sampling
)
add_warpx_test(
    test_3d_radiation_transport_group_sampling_cap
    3
    1
    "inputs_base_3d_radiation_transport_group_sampling radiation_transport.particle_conversion_max_packets_per_cell=2"
    "analysis_group_sampling.py --cap 2 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)

add_warpx_test(
    test_1d_radiation_transport_material_profile
    1
    1
    inputs_test_1d_radiation_transport_material_profile
    "analysis_material_profile.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)

add_warpx_test(
    test_1d_radiation_transport_material_profile_restart
    1
    1
    "inputs_test_1d_radiation_transport_material_profile_restart amr.restart='../test_1d_radiation_transport_material_profile/diags/chk000001'"
    "analysis_material_profile.py --restart --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    test_1d_radiation_transport_material_profile
)

if(WarpX_PRECISION STREQUAL "DOUBLE" AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_warpx_test(
        test_1d_radiation_transport_opposing_group_work
        1
        1
        inputs_test_1d_radiation_transport_opposing_group_work
        "analysis_opposing_group_work.py"
        OFF
        OFF
    )
    add_warpx_test(
        test_1d_radiation_transport_opposing_group_work_restart
        1
        2
        "inputs_test_1d_radiation_transport_opposing_group_work_restart amr.restart='../test_1d_radiation_transport_opposing_group_work/diags/chk000001'"
        "analysis_opposing_group_work.py --restart"
        OFF
        test_1d_radiation_transport_opposing_group_work
    )
endif()

add_warpx_test(
    test_3d_radiation_transport_energy_target_sampling
    3
    1
    inputs_test_3d_radiation_transport_energy_target_sampling
    "analysis_packet_sampling.py --packets-per-cell 4 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)
add_warpx_test(
    test_3d_radiation_transport_energy_target_sampling_cap
    3
    1
    inputs_test_3d_radiation_transport_energy_target_sampling_cap
    "analysis_packet_sampling.py --packets-per-cell 2 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)

add_warpx_test(
    test_3d_radiation_transport_energy_target_uniform
    3
    1
    inputs_test_3d_radiation_transport_energy_target_uniform
    "analysis_packet_sampling.py --packets-per-cell 4 --uniform-domain --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)
add_warpx_test(
    test_3d_radiation_transport_energy_target_uniform_mpi
    3
    2
    inputs_test_3d_radiation_transport_energy_target_uniform_mpi
    "analysis_packet_sampling.py --packets-per-cell 4 --uniform-domain --reference ../test_3d_radiation_transport_energy_target_uniform --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    test_3d_radiation_transport_energy_target_uniform
)

# Fixture commands below are registered outside add_warpx_test's function
# scope. Resolve their interpreter here as well, including APP-only builds.
if(WarpX_APP AND NOT Python_EXECUTABLE)
    find_package(Python COMPONENTS Interpreter REQUIRED)
endif()

if(WarpX_LIB)
    if("1" IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL "DOUBLE"
       AND (WarpX_COMPUTE STREQUAL "OMP" OR WarpX_COMPUTE STREQUAL "NOACC"))
        add_executable(radiation_stationary_restart_driver stationary_restart_driver.cpp)
        target_link_libraries(radiation_stationary_restart_driver PRIVATE lib_1d)
        target_compile_features(radiation_stationary_restart_driver PRIVATE cxx_std_20)
        warpx_set_compile_warnings(radiation_stationary_restart_driver)
    endif()
    if(WarpX_SINGULARITY_EOS AND "1" IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL "DOUBLE")
        add_executable(test_coupled_spiner test_coupled_spiner.cpp)
        target_link_libraries(test_coupled_spiner PRIVATE lib_1d)
        target_compile_features(test_coupled_spiner PRIVATE cxx_std_20)
        warpx_set_compile_warnings(test_coupled_spiner)
        set(coupled_spiner_table "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/radiation_coupled_spiner_fixture/electron.sp5")
        add_test(NAME radiation_coupled_spiner_fixture COMMAND ${CMAKE_COMMAND} -E env
            "WARPX_SPINER_FIXTURE_PATH=${coupled_spiner_table}"
            "WARPX_SPINER_FIXTURE_ATOMIC_MASS=13.019" "WARPX_SPINER_FIXTURE_ATOMIC_NUMBER=2"
            "WARPX_SPINER_FIXTURE_NUM_T=512" "WARPX_SPINER_FIXTURE_RHO_MAX=15"
            $<TARGET_FILE:test_hybrid_electron_thermodynamics_spiner>)
        set_tests_properties(radiation_coupled_spiner_fixture PROPERTIES FIXTURES_SETUP coupled_spiner_table
            ENVIRONMENT "OMP_NUM_THREADS=1")
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/coupled_spiner)
        add_test(NAME test_coupled_spiner COMMAND test_coupled_spiner inputs_base_1d_coupled_spiner
            hybrid_pic_model.electron_eos_ions_table_file=${coupled_spiner_table})
        set_tests_properties(test_coupled_spiner PROPERTIES FIXTURES_REQUIRED coupled_spiner_table
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/coupled_spiner
            ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
        if(WarpX_MPI)
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/coupled_spiner_mpi)
            add_test(NAME test_coupled_spiner_mpi COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_coupled_spiner> inputs_base_1d_coupled_spiner
                hybrid_pic_model.electron_eos_ions_table_file=${coupled_spiner_table}
                ${MPIEXEC_POSTFLAGS})
            set_tests_properties(test_coupled_spiner_mpi PROPERTIES FIXTURES_REQUIRED coupled_spiner_table
                PROCESSORS 2 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/coupled_spiner_mpi
                ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
        endif()
    endif()
    if(WarpX_PRECISION STREQUAL "DOUBLE")
        list(GET WarpX_DIMS 0 frame_dimension)
        warpx_set_suffix_dims(frame_suffix ${frame_dimension})
        add_executable(test_radiation_frame test_radiation_frame.cpp)
        target_link_libraries(test_radiation_frame PRIVATE lib_${frame_suffix})
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_radiation_frame)
            target_compile_features(test_radiation_frame PRIVATE cuda_std_20)
        else()
            target_compile_features(test_radiation_frame PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_radiation_frame)
        add_test(NAME test_radiation_frame COMMAND test_radiation_frame amrex.the_arena_init_size=0)
        set_tests_properties(test_radiation_frame PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
        add_executable(test_material_kinetic_work test_material_kinetic_work.cpp)
        target_link_libraries(test_material_kinetic_work PRIVATE lib_${frame_suffix})
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_material_kinetic_work)
            target_compile_features(test_material_kinetic_work PRIVATE cuda_std_20)
        else()
            target_compile_features(test_material_kinetic_work PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_material_kinetic_work)
        add_test(NAME test_material_kinetic_work COMMAND test_material_kinetic_work
            amrex.the_arena_init_size=0)
        set_tests_properties(test_material_kinetic_work PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
        add_executable(test_implicit_moment_source test_implicit_moment_source.cpp)
        target_link_libraries(test_implicit_moment_source PRIVATE lib_${frame_suffix})
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_implicit_moment_source)
            target_compile_features(test_implicit_moment_source PRIVATE cuda_std_20)
        else()
            target_compile_features(test_implicit_moment_source PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_implicit_moment_source)
        add_test(NAME test_implicit_moment_source COMMAND test_implicit_moment_source
            amrex.the_arena_init_size=0)
        set_tests_properties(test_implicit_moment_source PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
        add_executable(test_moving_moment_flux test_moving_moment_flux.cpp)
        target_link_libraries(test_moving_moment_flux PRIVATE lib_${frame_suffix})
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_moving_moment_flux)
            target_compile_features(test_moving_moment_flux PRIVATE cuda_std_20)
        else()
            target_compile_features(test_moving_moment_flux PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_moving_moment_flux)
        add_test(NAME test_moving_moment_flux COMMAND test_moving_moment_flux
            amrex.the_arena_init_size=0)
        set_tests_properties(test_moving_moment_flux PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
        add_executable(test_moment_boundary_flux test_moment_boundary_flux.cpp)
        # These are layout/reduction unit tests, not repeated production cases.
        foreach(boundary_dimension IN LISTS WarpX_DIMS)
            if(NOT boundary_dimension MATCHES "^[123]$")
                continue()
            endif()
            warpx_set_suffix_dims(boundary_suffix ${boundary_dimension})
            set(boundary_target test_moment_boundary_exchange_${boundary_suffix})
            add_executable(${boundary_target} test_moment_boundary_exchange.cpp)
            target_link_libraries(${boundary_target} PRIVATE lib_${boundary_suffix})
            if(WarpX_COMPUTE STREQUAL CUDA)
                setup_target_for_cuda_compilation(${boundary_target})
                target_compile_features(${boundary_target} PRIVATE cuda_std_20)
            else()
                target_compile_features(${boundary_target} PRIVATE cxx_std_20)
            endif()
            warpx_set_compile_warnings(${boundary_target})
            add_test(NAME ${boundary_target} COMMAND ${boundary_target} amrex.the_arena_init_size=0)
            set_tests_properties(${boundary_target} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
            if(WarpX_MPI)
                add_test(NAME ${boundary_target}_mpi COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:${boundary_target}> ${MPIEXEC_POSTFLAGS} amrex.the_arena_init_size=0)
                set_tests_properties(${boundary_target}_mpi PROPERTIES
                    ENVIRONMENT "OMP_NUM_THREADS=1" PROCESSORS 2)
            endif()
        endforeach()
        target_link_libraries(test_moment_boundary_flux PRIVATE lib_${frame_suffix})
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_moment_boundary_flux)
            target_compile_features(test_moment_boundary_flux PRIVATE cuda_std_20)
        else()
            target_compile_features(test_moment_boundary_flux PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_moment_boundary_flux)
        set(boundary_dir ${CMAKE_CURRENT_BINARY_DIR}/moment_boundary)
        file(MAKE_DIRECTORY ${boundary_dir})
        add_test(NAME test_moment_boundary_flux COMMAND test_moment_boundary_flux
            amrex.the_arena_init_size=0)
        set_tests_properties(test_moment_boundary_flux PROPERTIES
            WORKING_DIRECTORY ${boundary_dir} ENVIRONMENT "OMP_NUM_THREADS=1"
            FIXTURES_SETUP moment_boundary_data)
        if(NOT Python_EXECUTABLE)
            find_package(Python COMPONENTS Interpreter REQUIRED)
        endif()
        add_test(NAME test_moment_boundary_flux_analysis COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/reference_moment_boundary.py
            --kernel-output ${boundary_dir}/boundary_flux.txt)
        set_tests_properties(test_moment_boundary_flux_analysis PROPERTIES
            FIXTURES_REQUIRED moment_boundary_data)
        add_executable(test_implicit_moment_transport test_implicit_moment_transport.cpp)
        if(frame_dimension MATCHES "^[123]$")
            add_executable(test_particle_boundary_event test_particle_boundary_event.cpp)
            target_link_libraries(test_particle_boundary_event PRIVATE lib_${frame_suffix})
            if(WarpX_COMPUTE STREQUAL CUDA)
                setup_target_for_cuda_compilation(test_particle_boundary_event)
                target_compile_features(test_particle_boundary_event PRIVATE cuda_std_20)
            else()
                target_compile_features(test_particle_boundary_event PRIVATE cxx_std_20)
            endif()
            warpx_set_compile_warnings(test_particle_boundary_event)
            add_test(NAME test_particle_boundary_event COMMAND test_particle_boundary_event
                amrex.the_arena_init_size=0)
            set_tests_properties(test_particle_boundary_event PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
            add_executable(test_prescribed_moment_boundary test_prescribed_moment_boundary.cpp)
            target_link_libraries(test_prescribed_moment_boundary PRIVATE lib_${frame_suffix})
            if(WarpX_COMPUTE STREQUAL CUDA)
                setup_target_for_cuda_compilation(test_prescribed_moment_boundary)
                target_compile_features(test_prescribed_moment_boundary PRIVATE cuda_std_20)
            else()
                target_compile_features(test_prescribed_moment_boundary PRIVATE cxx_std_20)
            endif()
            warpx_set_compile_warnings(test_prescribed_moment_boundary)
            add_test(NAME test_prescribed_moment_boundary COMMAND test_prescribed_moment_boundary
                amrex.the_arena_init_size=0)
            set_tests_properties(test_prescribed_moment_boundary PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
            add_test(NAME test_prescribed_moment_boundary_outflow COMMAND test_prescribed_moment_boundary
                test.injection=-0.1 amrex.the_arena_init_size=0)
            set_tests_properties(test_prescribed_moment_boundary_outflow PROPERTIES
                ENVIRONMENT "OMP_NUM_THREADS=1")
            add_test(NAME test_reflecting_moment_boundary COMMAND test_prescribed_moment_boundary
                test.reflecting=1 amrex.the_arena_init_size=0)
            set_tests_properties(test_reflecting_moment_boundary PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
            if(WarpX_MPI)
                add_test(NAME test_prescribed_moment_boundary_mpi COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_prescribed_moment_boundary> ${MPIEXEC_POSTFLAGS}
                    amrex.the_arena_init_size=0)
                set_tests_properties(test_prescribed_moment_boundary_mpi PROPERTIES
                    ENVIRONMENT "OMP_NUM_THREADS=1" PROCESSORS 2)
                add_test(NAME test_reflecting_moment_boundary_mpi COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_prescribed_moment_boundary> ${MPIEXEC_POSTFLAGS}
                    test.reflecting=1 amrex.the_arena_init_size=0)
                set_tests_properties(test_reflecting_moment_boundary_mpi PROPERTIES
                    ENVIRONMENT "OMP_NUM_THREADS=1" PROCESSORS 2)
            endif()
        endif()
        target_link_libraries(test_implicit_moment_transport PRIVATE lib_${frame_suffix})
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_implicit_moment_transport)
            target_compile_features(test_implicit_moment_transport PRIVATE cuda_std_20)
        else()
            target_compile_features(test_implicit_moment_transport PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_implicit_moment_transport)
        if(frame_dimension STREQUAL "1")
            set(transport_refinement_dir "${CMAKE_CURRENT_BINARY_DIR}/moving_transport_refinement")
            file(MAKE_DIRECTORY ${transport_refinement_dir})
            add_test(NAME test_moving_transport_refinement COMMAND ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/analysis_moving_transport_refinement.py
                $<TARGET_FILE:test_implicit_moment_transport>)
            set_tests_properties(test_moving_transport_refinement PROPERTIES
                WORKING_DIRECTORY ${transport_refinement_dir} ENVIRONMENT "OMP_NUM_THREADS=1")
            set(cavity_refinement_dir "${CMAKE_CURRENT_BINARY_DIR}/reflecting_cavity")
            file(MAKE_DIRECTORY ${cavity_refinement_dir})
            add_test(NAME test_reflecting_cavity_refinement COMMAND ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/analysis_reflecting_cavity.py
                $<TARGET_FILE:test_implicit_moment_transport>)
            set_tests_properties(test_reflecting_cavity_refinement PROPERTIES
                WORKING_DIRECTORY ${cavity_refinement_dir} ENVIRONMENT "OMP_NUM_THREADS=1")
        endif()
    endif()
    foreach(D IN ITEMS 1 2 RZ)
        if(D IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL "DOUBLE")
            warpx_set_suffix_dims(SD ${D})
            if(NOT D STREQUAL "RZ" AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
                set(impulse_target test_${SD}_particle_impulse)
                add_executable(${impulse_target} test_particle_impulse.cpp)
                target_link_libraries(${impulse_target} PRIVATE lib_${SD})
                if(WarpX_COMPUTE STREQUAL CUDA)
                    setup_target_for_cuda_compilation(${impulse_target})
                    target_compile_features(${impulse_target} PRIVATE cuda_std_20)
                else()
                    target_compile_features(${impulse_target} PRIVATE cxx_std_20)
                endif()
                warpx_set_compile_warnings(${impulse_target})
                set(moment_target test_${SD}_coupled_moment_source)
                add_executable(${moment_target} test_coupled_moment_source.cpp)
                target_link_libraries(${moment_target} PRIVATE lib_${SD})
                if(WarpX_COMPUTE STREQUAL CUDA)
                    setup_target_for_cuda_compilation(${moment_target})
                    target_compile_features(${moment_target} PRIVATE cuda_std_20)
                else()
                    target_compile_features(${moment_target} PRIVATE cxx_std_20)
                endif()
                warpx_set_compile_warnings(${moment_target})
                foreach(source_kind IN ITEMS drag thermal latent stiff cooling interval
                                             interval_thermal
                                             gradient gradient_latent gradient_stiff
                                             spatial_drag spatial_thermal
                                             gradient_spatial gradient_spatial_latent
                                             gradient_spatial_stiff gradient_spatial_unit_rate interval_spatial
                                             interval_thermal_spatial gradient_shape gradient_shape_spatial)
                    if((source_kind STREQUAL "gradient_shape" AND NOT D STREQUAL "1") OR
                       (source_kind STREQUAL "gradient_shape_spatial" AND NOT D STREQUAL "2"))
                        continue()
                    endif()
                    set(moment_name ${moment_target}_${source_kind})
                    set(analysis_arguments)
                    set(source_arguments test.kind=${source_kind})
                    if(source_kind STREQUAL "spatial_drag")
                        set(source_arguments test.kind=drag)
                    endif()
                    if(source_kind STREQUAL "spatial_thermal")
                        set(source_arguments test.kind=thermal)
                    endif()
                    if(source_kind STREQUAL "interval" OR source_kind STREQUAL "interval_spatial")
                        set(source_arguments test.kind=drag test.interval=1)
                    endif()
                    if(source_kind MATCHES "^interval_thermal")
                        set(source_arguments test.kind=thermal test.interval=1)
                    endif()
                    if(source_kind STREQUAL "stiff")
                        set(source_arguments test.kind=thermal test.absorption_step=100)
                    endif()
                    if(source_kind STREQUAL "cooling")
                        set(source_arguments test.kind=thermal test.absorption_step=100
                            "hybrid_pic_model.initial_elec_temp(x,y,z)=10")
                    endif()
                    if(source_kind STREQUAL "latent" OR source_kind MATCHES "^gradient.*_latent$")
                        set(source_arguments test.kind=thermal
                            hybrid_pic_model.electron_thermodynamics=fixed_charge_latent_energy
                            hybrid_pic_model.electron_latent_transition_temperature_eV=1.0
                            hybrid_pic_model.electron_latent_energy_eV=2.0
                            hybrid_pic_model.electron_latent_sharpness=4.0)
                    endif()
                    if(source_kind MATCHES "^gradient")
                        if(NOT source_kind MATCHES "_latent$")
                            set(source_arguments test.kind=thermal)
                        endif()
                        list(APPEND source_arguments test.nonuniform=1)
                        if(source_kind MATCHES "_stiff$")
                            list(APPEND source_arguments test.absorption_step=100)
                        endif()
                        if(source_kind MATCHES "_unit_rate$")
                            list(APPEND source_arguments test.absorption_step=1)
                        endif()
                    endif()
                    if(source_kind MATCHES "spatial")
                        list(APPEND source_arguments test.spatial=1)
                        if(WarpX_COMPUTE STREQUAL CUDA)
                            list(APPEND source_arguments test.progress_interval=25)
                        endif()
                    endif()
                    if(source_kind MATCHES "shape")
                        list(APPEND source_arguments test.shape=1)
                        list(APPEND analysis_arguments --shaped)
                    endif()
                    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${moment_name})
                    add_test(NAME ${moment_name} COMMAND ${moment_target}
                        inputs_${SD}_coupled_moment_source ${source_arguments})
                    set_tests_properties(${moment_name} PROPERTIES
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${moment_name}
                        ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                    if(source_kind MATCHES "^gradient")
                        set_tests_properties(${moment_name} PROPERTIES FIXTURES_SETUP ${moment_name})
                        add_test(NAME ${moment_name}_analysis COMMAND ${Python_EXECUTABLE}
                            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_nonuniform_moment_source.py
                            ${CMAKE_CURRENT_BINARY_DIR}/${moment_name}/nonuniform_state.txt
                            ${analysis_arguments})
                        set_tests_properties(${moment_name}_analysis PROPERTIES
                            FIXTURES_REQUIRED ${moment_name})
                    endif()
                    if(WarpX_MPI)
                        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${moment_name}_mpi)
                        add_test(NAME ${moment_name}_mpi COMMAND
                            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                            $<TARGET_FILE:${moment_target}> inputs_${SD}_coupled_moment_source
                            ${source_arguments} ${MPIEXEC_POSTFLAGS})
                        set_tests_properties(${moment_name}_mpi PROPERTIES PROCESSORS 2
                            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${moment_name}_mpi
                            ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                        if(source_kind MATCHES "^gradient")
                            set_tests_properties(${moment_name}_mpi PROPERTIES
                                FIXTURES_SETUP ${moment_name}_mpi)
                            add_test(NAME ${moment_name}_mpi_analysis COMMAND ${Python_EXECUTABLE}
                                ${CMAKE_CURRENT_SOURCE_DIR}/analysis_nonuniform_moment_source.py
                                ${CMAKE_CURRENT_BINARY_DIR}/${moment_name}_mpi/nonuniform_state.txt
                                ${analysis_arguments})
                            set_tests_properties(${moment_name}_mpi_analysis PROPERTIES
                                FIXTURES_REQUIRED ${moment_name}_mpi)
                        endif()
                    endif()
                endforeach()
                set(impulse_dir "${CMAKE_CURRENT_BINARY_DIR}/particle_impulse_${SD}")
                # Complementary sustained shaped-coupling cases, not every
                # source variant repeated in every geometry.
                set(shape_kind drag)
                if(D STREQUAL "2")
                    set(shape_kind thermal)
                endif()
                set(shape_name ${moment_target}_shape_spatial)
                file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${shape_name})
                add_test(NAME ${shape_name} COMMAND ${moment_target}
                    inputs_${SD}_coupled_moment_source test.kind=${shape_kind}
                    test.shape=1 test.spatial=1 test.progress_interval=25)
                set_tests_properties(${shape_name} PROPERTIES
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${shape_name}
                    ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                file(MAKE_DIRECTORY ${impulse_dir})
                if(D STREQUAL "1")
                    foreach(carry_procs IN ITEMS 1 2)
                        if(carry_procs EQUAL 2 AND NOT WarpX_MPI)
                            continue()
                        endif()
                        set(carry_name ${impulse_target}_boundary_carry_${carry_procs})
                        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${carry_name})
                        set(carry_launcher)
                        set(carry_postflags)
                        if(carry_procs EQUAL 2)
                            set(carry_launcher ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                                ${MPIEXEC_PREFLAGS})
                            set(carry_postflags ${MPIEXEC_POSTFLAGS})
                        endif()
                        add_test(NAME ${carry_name} COMMAND ${carry_launcher}
                            $<TARGET_FILE:${impulse_target}> inputs_base_1d_particle_impulse
                            test.boundary_carry=1 boundary.field_lo=pec boundary.field_hi=pec
                            boundary.particle_lo=reflecting boundary.particle_hi=reflecting
                            ${carry_postflags})
                        set_tests_properties(${carry_name} PROPERTIES
                            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${carry_name}
                            PROCESSORS ${carry_procs}
                            ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                    endforeach()
                    if(WarpX_MPI)
                        set(carry_restart_name ${impulse_target}_boundary_carry_restart)
                        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${carry_restart_name})
                        add_test(NAME ${carry_restart_name} COMMAND
                            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                            $<TARGET_FILE:${impulse_target}> inputs_base_1d_particle_impulse
                            test.boundary_carry=1
                            test.boundary_carry_restart=${CMAKE_CURRENT_BINARY_DIR}/${impulse_target}_boundary_carry_1/compensated_carry_wall_checkpoint
                            boundary.field_lo=pec boundary.field_hi=pec
                            boundary.particle_lo=reflecting boundary.particle_hi=reflecting
                            ${MPIEXEC_POSTFLAGS})
                        set_tests_properties(${impulse_target}_boundary_carry_1 PROPERTIES
                            FIXTURES_SETUP carry_wall_checkpoint)
                        set_tests_properties(${carry_restart_name} PROPERTIES
                            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${carry_restart_name}
                            PROCESSORS 2 FIXTURES_REQUIRED carry_wall_checkpoint
                            ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                    endif()
                    foreach(wall_order IN ITEMS 1 2 3 4)
                        set(wall_variants normal)
                        if(wall_order EQUAL 4)
                            list(APPEND wall_variants tangent)
                            if(WarpX_MPI)
                                list(APPEND wall_variants mpi)
                            endif()
                        endif()
                        foreach(wall_variant IN LISTS wall_variants)
                            set(wall_name ${impulse_target}_reflecting_${wall_order}_${wall_variant})
                            set(wall_dir ${CMAKE_CURRENT_BINARY_DIR}/${wall_name})
                            file(MAKE_DIRECTORY ${wall_dir})
                            set(wall_component 2)
                            set(wall_launcher)
                            set(wall_postflags)
                            set(wall_procs 1)
                            if(wall_variant STREQUAL tangent)
                                set(wall_component 0)
                            elseif(wall_variant STREQUAL mpi)
                                set(wall_procs 2)
                                set(wall_launcher ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                                    ${MPIEXEC_PREFLAGS})
                                set(wall_postflags ${MPIEXEC_POSTFLAGS})
                            endif()
                            add_test(NAME ${wall_name} COMMAND ${wall_launcher}
                                $<TARGET_FILE:${impulse_target}> inputs_base_1d_particle_impulse
                                test.shape=1 test.reflecting=1 test.component=${wall_component}
                                algo.particle_shape=${wall_order}
                                boundary.field_lo=pec boundary.field_hi=pec
                                boundary.particle_lo=reflecting boundary.particle_hi=reflecting
                                ${wall_postflags})
                            set_tests_properties(${wall_name} PROPERTIES
                                WORKING_DIRECTORY ${wall_dir} PROCESSORS ${wall_procs}
                                ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                        endforeach()
                    endforeach()
                    set(wall_rejection_dir ${CMAKE_CURRENT_BINARY_DIR}/reflecting_rejects_periodic)
                    file(MAKE_DIRECTORY ${wall_rejection_dir})
                    add_test(NAME ${impulse_target}_reflecting_rejects_periodic COMMAND
                        ${Python_EXECUTABLE}
                        ${WarpX_SOURCE_DIR}/Examples/Tests/implicit/check_command_output.py
                        --expect-failure --expect "Reflecting nodal assignment requires 1D"
                        -- $<TARGET_FILE:${impulse_target}> inputs_base_1d_particle_impulse
                        test.shape=1 test.reflecting=1 amrex.throw_exception=1)
                    set_tests_properties(${impulse_target}_reflecting_rejects_periodic PROPERTIES
                        WORKING_DIRECTORY ${wall_rejection_dir}
                        ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                    set(immutable_dir "${CMAKE_CURRENT_BINARY_DIR}/particle_plot_immutable")
                    file(MAKE_DIRECTORY ${immutable_dir})
                    add_test(NAME ${impulse_target}_plot_immutable COMMAND ${impulse_target}
                        inputs_base_1d_particle_impulse test.plot_immutable=1)
                    set_tests_properties(${impulse_target}_plot_immutable PROPERTIES
                        FIXTURES_SETUP particle_plot_immutable
                        WORKING_DIRECTORY ${immutable_dir}
                        ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                    add_test(NAME ${impulse_target}_plot_immutable_analysis COMMAND
                        ${Python_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/analysis_particle_plot_immutable.py ${immutable_dir})
                    set_tests_properties(${impulse_target}_plot_immutable_analysis PROPERTIES
                        FIXTURES_REQUIRED particle_plot_immutable)
                    if(WarpX_OPENPMD)
                        set(openpmd_immutable_dir "${CMAKE_CURRENT_BINARY_DIR}/particle_openpmd_immutable")
                        file(MAKE_DIRECTORY ${openpmd_immutable_dir})
                        add_test(NAME ${impulse_target}_openpmd_immutable COMMAND ${impulse_target}
                            inputs_base_1d_particle_impulse test.plot_immutable=1 test.openpmd=1
                            readonly.diag_type=Full readonly.openpmd_backend=json)
                        set_tests_properties(${impulse_target}_openpmd_immutable PROPERTIES
                            FIXTURES_SETUP particle_openpmd_immutable
                            WORKING_DIRECTORY ${openpmd_immutable_dir}
                            ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                        add_test(NAME ${impulse_target}_openpmd_immutable_analysis COMMAND
                            ${Python_EXECUTABLE}
                            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_particle_plot_immutable.py
                            ${immutable_dir} --openpmd ${openpmd_immutable_dir})
                        set_tests_properties(${impulse_target}_openpmd_immutable_analysis PROPERTIES
                            FIXTURES_REQUIRED "particle_plot_immutable;particle_openpmd_immutable")
                    endif()
                endif()
                set(birth_carry_dir ${impulse_dir}/birth_carry)
                file(MAKE_DIRECTORY ${birth_carry_dir})
                add_test(NAME ${impulse_target}_birth_carry COMMAND ${impulse_target}
                    inputs_base_${SD}_particle_impulse test.birth_carry=1)
                set_tests_properties(${impulse_target}_birth_carry PROPERTIES
                    WORKING_DIRECTORY ${birth_carry_dir}
                    ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                add_test(NAME ${impulse_target}_shape COMMAND ${impulse_target}
                    inputs_base_${SD}_particle_impulse test.shape=1)
                set_tests_properties(${impulse_target}_shape PROPERTIES
                    WORKING_DIRECTORY ${impulse_dir}
                    ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                foreach(shape_order IN ITEMS 2 3 4)
                    add_test(NAME ${impulse_target}_shape${shape_order} COMMAND ${impulse_target}
                        inputs_base_${SD}_particle_impulse test.shape=1 algo.particle_shape=${shape_order})
                    set_tests_properties(${impulse_target}_shape${shape_order} PROPERTIES
                        WORKING_DIRECTORY ${impulse_dir}
                        ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                    if(WarpX_MPI)
                        add_test(NAME ${impulse_target}_shape${shape_order}_mpi COMMAND
                            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                            $<TARGET_FILE:${impulse_target}> inputs_base_${SD}_particle_impulse
                            test.shape=1 algo.particle_shape=${shape_order} ${MPIEXEC_POSTFLAGS})
                        set_tests_properties(${impulse_target}_shape${shape_order}_mpi PROPERTIES
                            PROCESSORS 2 WORKING_DIRECTORY ${impulse_dir}
                            ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                    endif()
                endforeach()
                add_test(NAME ${impulse_target} COMMAND ${impulse_target}
                    inputs_base_${SD}_particle_impulse test.checkpoint=${impulse_dir}/checkpoint)
                set_tests_properties(${impulse_target} PROPERTIES
                    FIXTURES_SETUP impulse_checkpoint_${SD} WORKING_DIRECTORY ${impulse_dir}
                    ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                if(D STREQUAL "2")
                    add_test(NAME ${impulse_target}_shape_ownership COMMAND ${impulse_target}
                        inputs_base_${SD}_particle_impulse test.shape_ownership=1
                        test.checkpoint=${impulse_dir}/shape_checkpoint)
                    set_tests_properties(${impulse_target}_shape_ownership PROPERTIES
                        FIXTURES_SETUP shape_checkpoint_${SD} WORKING_DIRECTORY ${impulse_dir}
                        ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                    if(WarpX_MPI)
                        add_test(NAME ${impulse_target}_shape_restart_mpi COMMAND
                            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                            $<TARGET_FILE:${impulse_target}> inputs_base_${SD}_particle_impulse
                            test.shape_ownership=1 test.restart=${impulse_dir}/shape_checkpoint
                            ${MPIEXEC_POSTFLAGS})
                        set_tests_properties(${impulse_target}_shape_restart_mpi PROPERTIES
                            FIXTURES_REQUIRED shape_checkpoint_${SD} PROCESSORS 2
                            WORKING_DIRECTORY ${impulse_dir}
                            ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                    endif()
                endif()
                if(WarpX_MPI)
                    add_test(NAME ${impulse_target}_shape_mpi COMMAND
                        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:${impulse_target}> inputs_base_${SD}_particle_impulse
                        test.shape=1 ${MPIEXEC_POSTFLAGS})
                    set_tests_properties(${impulse_target}_shape_mpi PROPERTIES
                        PROCESSORS 2 WORKING_DIRECTORY ${impulse_dir}
                        ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                    add_test(NAME ${impulse_target}_restart_mpi COMMAND
                        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:${impulse_target}> inputs_base_${SD}_particle_impulse
                        test.restart=${impulse_dir}/checkpoint ${MPIEXEC_POSTFLAGS})
                    set_tests_properties(${impulse_target}_restart_mpi PROPERTIES
                        FIXTURES_REQUIRED impulse_checkpoint_${SD} PROCESSORS 2
                        WORKING_DIRECTORY ${impulse_dir}
                        ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                endif()
            endif()
            set(extension_target test_${SD}_coupled_material_extensions)
            add_executable(${extension_target} test_coupled_material_extensions.cpp)
            target_link_libraries(${extension_target} PRIVATE lib_${SD})
            if(WarpX_COMPUTE STREQUAL CUDA)
                setup_target_for_cuda_compilation(${extension_target})
                target_compile_features(${extension_target} PRIVATE cuda_std_20)
            else()
                target_compile_features(${extension_target} PRIVATE cxx_std_20)
            endif()
            warpx_set_compile_warnings(${extension_target})
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${extension_target})
            add_test(NAME ${extension_target} COMMAND ${extension_target}
                inputs_base_${SD}_coupled_material_extensions)
            set_tests_properties(${extension_target} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${extension_target}
                ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${extension_target}_tables)
            add_test(NAME ${extension_target}_tables COMMAND ${extension_target}
                inputs_base_${SD}_coupled_material_extensions
                test.opacity_table_root=${CMAKE_CURRENT_SOURCE_DIR})
            set_tests_properties(${extension_target}_tables PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${extension_target}_tables
                ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
            if(WarpX_MPI)
                file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${extension_target}_mpi)
                add_test(NAME ${extension_target}_mpi COMMAND
                    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:${extension_target}>
                    inputs_base_${SD}_coupled_material_extensions ${MPIEXEC_POSTFLAGS})
                set_tests_properties(${extension_target}_mpi PROPERTIES PROCESSORS 2
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${extension_target}_mpi
                    ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
            endif()
        endif()
    endforeach()
    if("3" IN_LIST WarpX_DIMS)
        add_executable(test_radiation_diffusion_gradient test_diffusion_gradient.cpp)
        target_include_directories(test_radiation_diffusion_gradient PRIVATE ${WarpX_SOURCE_DIR}/Source)
        target_link_libraries(test_radiation_diffusion_gradient PRIVATE AMReX::amrex_3d)
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_radiation_diffusion_gradient)
            target_compile_features(test_radiation_diffusion_gradient PRIVATE cuda_std_20)
        else()
            target_compile_features(test_radiation_diffusion_gradient PRIVATE cxx_std_20)
        endif()
        add_test(NAME test_radiation_diffusion_gradient
            COMMAND test_radiation_diffusion_gradient amrex.the_arena_init_size=0)
        set_tests_properties(test_radiation_diffusion_gradient PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
    if("RZ" IN_LIST WarpX_DIMS AND NOT WarpX_COMPUTE STREQUAL CUDA)
        add_executable(test_radiation_implicit_diffusion_rz
            test_implicit_diffusion_rz.cpp)
        target_link_libraries(test_radiation_implicit_diffusion_rz PRIVATE lib_rz)
        target_compile_features(test_radiation_implicit_diffusion_rz PRIVATE cxx_std_20)
        warpx_set_compile_warnings(test_radiation_implicit_diffusion_rz)
        add_test(NAME test_radiation_implicit_diffusion_rz COMMAND test_radiation_implicit_diffusion_rz)
        set_tests_properties(test_radiation_implicit_diffusion_rz PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
    if("1" IN_LIST WarpX_DIMS AND NOT WarpX_COMPUTE STREQUAL CUDA)
        if(WarpX_PRECISION STREQUAL "DOUBLE")
            add_executable(test_hybrid_material_response test_hybrid_material_response.cpp)
            target_link_libraries(test_hybrid_material_response PRIVATE lib_1d)
            target_compile_features(test_hybrid_material_response PRIVATE cxx_std_20)
            warpx_set_compile_warnings(test_hybrid_material_response)
            add_test(NAME test_hybrid_material_response
                COMMAND test_hybrid_material_response inputs_base_1d_hybrid_material_response)
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/material_response)
            set_tests_properties(test_hybrid_material_response PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/material_response
                ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
            if(WarpX_MPI)
                file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/material_response_mpi)
                add_test(NAME test_hybrid_material_response_mpi COMMAND
                    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_hybrid_material_response>
                    inputs_base_1d_hybrid_material_response ${MPIEXEC_POSTFLAGS})
                set_tests_properties(test_hybrid_material_response_mpi PROPERTIES
                    PROCESSORS 2 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/material_response_mpi
                    ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
            endif()
        endif()
        add_executable(test_radiation_implicit_diffusion
            test_implicit_diffusion.cpp)
        target_link_libraries(test_radiation_implicit_diffusion PRIVATE lib_1d)
        target_compile_features(test_radiation_implicit_diffusion PRIVATE cxx_std_20)
        warpx_set_compile_warnings(test_radiation_implicit_diffusion)
        add_test(NAME test_radiation_implicit_diffusion COMMAND test_radiation_implicit_diffusion)
        set_tests_properties(test_radiation_implicit_diffusion PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
    add_executable(
        test_radiation_energy_update
        test_radiation_energy_update.cpp)
    target_link_libraries(
        test_radiation_energy_update
        PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_radiation_energy_update)
        target_compile_features(
            test_radiation_energy_update PRIVATE cuda_std_20)
        set_target_properties(
            test_radiation_energy_update PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON)
    else()
        target_compile_features(
            test_radiation_energy_update PRIVATE cxx_std_20)
        set_target_properties(
            test_radiation_energy_update PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON)
    endif()
    warpx_set_compile_warnings(test_radiation_energy_update)

    add_test(
        NAME test_radiation_energy_update
        COMMAND test_radiation_energy_update)

    add_executable(
        test_radiation_kinetic_energy_update
        test_radiation_kinetic_energy_update.cpp)
    target_link_libraries(
        test_radiation_kinetic_energy_update
        PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_radiation_kinetic_energy_update)
        target_compile_features(
            test_radiation_kinetic_energy_update PRIVATE cuda_std_20)
        set_target_properties(
            test_radiation_kinetic_energy_update PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON)
    else()
        target_compile_features(
            test_radiation_kinetic_energy_update PRIVATE cxx_std_20)
        set_target_properties(
            test_radiation_kinetic_energy_update PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON)
    endif()
    warpx_set_compile_warnings(test_radiation_kinetic_energy_update)

    add_test(
        NAME test_radiation_kinetic_energy_update
        COMMAND test_radiation_kinetic_energy_update)

    add_executable(
        test_radial_face_marching
        test_radial_face_marching.cpp)
    target_link_libraries(
        test_radial_face_marching
        PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_radial_face_marching)
        target_compile_features(
            test_radial_face_marching PRIVATE cuda_std_20)
        set_target_properties(
            test_radial_face_marching PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON)
    else()
        target_compile_features(
            test_radial_face_marching PRIVATE cxx_std_20)
        set_target_properties(
            test_radial_face_marching PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON)
    endif()
    warpx_set_compile_warnings(test_radial_face_marching)

    add_test(
        NAME test_radial_face_marching
        COMMAND test_radial_face_marching)
endif()

# CTest's PASS_REGULAR_EXPRESSION does not require a failed child process. Use
# the shared command-contract wrapper so unsupported configurations must both
# fail and emit their intended diagnostic before accepting a timestep. An
# optional fifth argument requests an MPI rank count; the default is one.
function(add_radiation_failure_test name dims inputs expected)
    set(nprocs 1)
    if(ARGC GREATER 4)
        set(nprocs ${ARGV4})
    endif()
    if(nprocs GREATER_EQUAL 2 AND NOT WarpX_MPI)
        return()
    endif()
    if(NOT ${dims} IN_LIST WarpX_DIMS OR NOT WarpX_APP)
        return()
    endif()
    if(NOT Python_EXECUTABLE)
        find_package(Python COMPONENTS Interpreter REQUIRED)
    endif()
    separate_arguments(input_args UNIX_COMMAND "${inputs}")
    warpx_set_suffix_dims(SD ${dims})
    set(test_working_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}")
    file(MAKE_DIRECTORY "${test_working_dir}")
    set(test_mpi_command "")
    if(nprocs GREATER_EQUAL 2)
        set(test_mpi_command
            ${MPIEXEC_EXECUTABLE}
            ${MPI_ALLOW_ROOT}
            ${MPIEXEC_NUMPROC_FLAG} ${nprocs}
            ${MPIEXEC_POSTFLAGS}
            ${MPIEXEC_PREFLAGS})
    endif()
    add_test(
        NAME ${name}.run
        COMMAND
            ${Python_EXECUTABLE}
            ${WarpX_SOURCE_DIR}/Examples/Tests/implicit/check_command_output.py
            --expect-failure
            --expect "${expected}"
            --forbid "STEP 1 ends"
            --
            ${test_mpi_command}
            $<TARGET_FILE:app_${SD}>
            ${input_args}
            "amrex.abort_on_unused_inputs = 1"
            "amrex.throw_exception = 1"
            "amrex.the_arena_init_size = 0"
            "warpx.always_warn_immediately = 1"
            "warpx.do_dynamic_scheduling = 0"
            "warpx.serialize_initial_conditions = 1"
        WORKING_DIRECTORY "${test_working_dir}")
    set_property(
        TEST ${name}.run APPEND PROPERTY
        ENVIRONMENT
            "AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/"
            "OMP_NUM_THREADS=1")
endfunction()

if(WarpX_PRECISION STREQUAL DOUBLE)
    if(WarpX_PARTICLE_PRECISION STREQUAL DOUBLE)
        add_radiation_failure_test(
            test_1d_radiation_transport_material_wall_requires_work
            1 "inputs_base_1d_material_wall_control hybrid_pic_model.conservative_pressure_work_pec=1 hybrid_pic_model.conservative_pressure_work=0"
            "PEC pressure work requires conservative_pressure_work=1")
        add_radiation_failure_test(
            test_1d_radiation_transport_material_wall_rejects_particle_loss
            1 "inputs_base_1d_material_wall_control hybrid_pic_model.conservative_pressure_work_pec=1 boundary.particle_hi=absorbing"
            "conservative_pressure_work_pec requires stationary PEC fields")
        add_radiation_failure_test(
            test_1d_radiation_transport_moment_inventory_mode_rejected
            1 "inputs_base_1d_moving_moment_beam radiation_transport.diffusion_solver=explicit"
            "include_moment_inventory requires coupled_moment")
        add_radiation_failure_test(
            test_1d_radiation_transport_moving_momentum_schema_rejected
            1 "inputs_base_1d_moving_moment_beam amr.restart='../test_1d_radiation_transport_moving_beam_ode/diags/chk000200' radiation_momentum.include_moment_inventory=0"
            "RadiationMomentum restart must preserve include_moment_inventory")
        if(TEST test_1d_radiation_transport_moving_momentum_schema_rejected.run)
            set_property(TEST test_1d_radiation_transport_moving_beam_ode.run
                APPEND PROPERTY FIXTURES_SETUP radiation_moving_momentum_checkpoint)
            set_property(TEST test_1d_radiation_transport_moving_momentum_schema_rejected.run
                APPEND PROPERTY FIXTURES_REQUIRED radiation_moving_momentum_checkpoint)
            if(TEST test_1d_radiation_transport_moving_beam_ode.cleanup)
                set_property(TEST test_1d_radiation_transport_moving_beam_ode.cleanup
                    APPEND PROPERTY FIXTURES_CLEANUP radiation_moving_momentum_checkpoint)
            endif()
        endif()
        add_radiation_failure_test(
            test_1d_radiation_transport_moving_moment_shape_restart_rejected
            1 "inputs_base_1d_moving_moment_pulse max_step=400 amr.n_cell=128 amr.max_grid_size=64 warpx.const_dt=8.333333333333333e-12 chk.intervals=200 algo.particle_shape=3 amr.restart='../test_1d_radiation_transport_moving_moment_pulse/diags/chk000200'"
            "Moving radiation restart must preserve its particle shape order.")
        if(TEST test_1d_radiation_transport_moving_moment_shape_restart_rejected.run)
            set_property(TEST test_1d_radiation_transport_moving_moment_pulse.run
                APPEND PROPERTY FIXTURES_SETUP radiation_moving_moment_checkpoint)
            set_property(TEST test_1d_radiation_transport_moving_moment_shape_restart_rejected.run
                APPEND PROPERTY FIXTURES_REQUIRED radiation_moving_moment_checkpoint)
            if(TEST test_1d_radiation_transport_moving_moment_pulse.cleanup)
                set_property(TEST test_1d_radiation_transport_moving_moment_pulse.cleanup
                    APPEND PROPERTY FIXTURES_CLEANUP radiation_moving_moment_checkpoint)
            endif()
            foreach(missing IN ITEMS model x y z)
                set(corrupt_name test_1d_radiation_transport_moving_moment_missing_${missing})
                set(corrupt_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${corrupt_name})
                file(MAKE_DIRECTORY ${corrupt_dir})
                add_test(NAME ${corrupt_name} COMMAND ${Python_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/check_moving_moment_checkpoint.py
                    $<TARGET_FILE:app_1d>
                    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_1d_radiation_transport_moving_moment_pulse/diags/chk000200
                    ${missing})
                set_tests_properties(${corrupt_name} PROPERTIES
                    WORKING_DIRECTORY ${corrupt_dir}
                    FIXTURES_REQUIRED radiation_moving_moment_checkpoint
                    TIMEOUT 90
                    ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
            endforeach()
        endif()
    endif()
    add_radiation_failure_test(
        test_1d_radiation_transport_coupled_moving_rejected
        1 "inputs_base_1d_radiation_transport_coupled_implicit ions.do_not_push=0"
        "coupled_implicit currently requires stationary")
    add_radiation_failure_test(
        test_1d_radiation_transport_coupled_packets_rejected
        1 "inputs_base_1d_radiation_transport_coupled_implicit photons.injection_style=SingleParticle photons.single_particle_pos=0 0 0.42 photons.single_particle_u=0 0 1 photons.single_particle_weight=1"
        "coupled_implicit currently requires no live")
    add_radiation_failure_test(
        test_1d_radiation_transport_coupled_budget_rejected
        1 "inputs_base_1d_radiation_transport_coupled_implicit radiation_transport.lte_exchange_max_iterations=1"
        "Coupled implicit radiation/material solve failed (reason 3)")
    add_radiation_failure_test(
        test_1d_radiation_transport_coupled_conversion_rejected
        1 "inputs_base_1d_radiation_transport_coupled_implicit radiation_transport.enable_particle_conversion=1 radiation_transport.emission_photon_energy=1.e-17"
        "Coupled implicit diffusion currently requires")
    add_radiation_failure_test(
        test_1d_radiation_transport_coupled_momentum_rejected
        1 "inputs_base_1d_radiation_transport_coupled_implicit radiation_transport.enable_momentum_coupling=1 radiation_transport.momentum_species=ions"
        "Coupled implicit diffusion currently requires")
    add_radiation_failure_test(
        test_1d_radiation_transport_implicit_periodic_bath_rejected
        1
        "inputs_base_1d_radiation_transport_bath radiation_transport.diffusion_solver=implicit boundary.field_lo=periodic boundary.field_hi=periodic boundary.particle_lo=periodic boundary.particle_hi=periodic"
        "Implicit radiation bath faces must be nonperiodic")
endif()

add_radiation_failure_test(
    test_1d_radiation_transport_initial_blackbody_negative_rejected
    1
    "inputs_base_1d_radiation_transport_initial_blackbody radiation_transport.initial_diffusion_temperature(x,y,z)=-1.0"
    "negative or non-finite")

add_radiation_failure_test(
    test_1d_radiation_transport_implicit_lte_rejected
    1
    "inputs_base_1d_radiation_transport_bath radiation_transport.diffusion_solver=implicit radiation_transport.enable_lte_exchange=1 radiation_transport.planck_absorption_coefficient=0.0"
    "transport-only feature")

add_radiation_failure_test(
    test_1d_radiation_transport_bath_negative_rejected
    1
    "inputs_base_1d_radiation_transport_bath radiation_transport.diffusion_bath_energy_density_lo_0_g0(x,y,z,t)=-1.0"
    "Radiation diffusion failed validation (reason 2")
add_radiation_failure_test(
    test_1d_radiation_transport_bath_thin_rejected
    1
    "inputs_base_1d_radiation_transport_bath radiation_transport.rosseland_transport_coefficient=0.5"
    "Radiation diffusion failed validation (reason 3")
add_radiation_failure_test(
    test_1d_radiation_transport_bath_conversion_rejected
    1
    "inputs_base_1d_radiation_transport_bath radiation_transport.enable_particle_conversion=1"
    "marshak_bath currently requires diffusion on a fixed grid")

add_radiation_failure_test(
    test_1d_radiation_transport_bath_removal_restart_rejected
    1
    "inputs_base_1d_radiation_transport_bath_common radiation_transport.diffusion_boundary_lo=marshak radiation_transport.diffusion_boundary_hi=marshak amr.restart='../test_1d_radiation_transport_bath/diags/chk000050'"
    "A restart with a bath injection ledger")
if(WarpX_PRECISION STREQUAL "DOUBLE")
    add_radiation_failure_test(
        test_1d_radiation_transport_coupled_details_removal_rejected
        1 "inputs_base_1d_radiation_transport_coupled_implicit amr.restart='../test_1d_radiation_transport_coupled_retry_details/diags/chk000010'"
        "solver-details checkpoint configuration changed")
    if(WarpX_APP AND "1" IN_LIST WarpX_DIMS)
        set_property(TEST test_1d_radiation_transport_coupled_retry_details.run
            APPEND PROPERTY FIXTURES_SETUP radiation_solver_details_checkpoint)
        set_property(TEST test_1d_radiation_transport_coupled_details_removal_rejected.run
            APPEND PROPERTY FIXTURES_REQUIRED radiation_solver_details_checkpoint)
        if(TEST test_1d_radiation_transport_coupled_retry_details.cleanup)
            set_property(TEST test_1d_radiation_transport_coupled_retry_details.cleanup
                APPEND PROPERTY FIXTURES_CLEANUP radiation_solver_details_checkpoint)
        endif()
    endif()
endif()
if(WarpX_APP AND "1" IN_LIST WarpX_DIMS)
    # A run dependency alone does not keep the checkpoint alive until this
    # rejection test: CI cleanup can otherwise remove it after the valid restart.
    # A fixture also selects the producer when running only the rejection test.
    set_property(TEST test_1d_radiation_transport_bath.run APPEND PROPERTY
        FIXTURES_SETUP radiation_bath_checkpoint)
    set_property(TEST test_1d_radiation_transport_bath_removal_restart_rejected.run
        APPEND PROPERTY FIXTURES_REQUIRED radiation_bath_checkpoint)
    if(TEST test_1d_radiation_transport_bath.cleanup)
        set_property(TEST test_1d_radiation_transport_bath.cleanup APPEND PROPERTY
            FIXTURES_CLEANUP radiation_bath_checkpoint)
    endif()
endif()

add_warpx_test(
    test_1d_radiation_transport_beer_lambert  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_beer_lambert  # inputs
    "analysis_beer_lambert.py --geometry 1d --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_small_constant_opacity
    1
    1
    inputs_test_1d_radiation_transport_small_constant_opacity
    "analysis_beer_lambert.py --geometry 1d --length-scale 1.0e8 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
    OFF
    OFF
)

add_warpx_test(
    test_1d_radiation_transport_beer_lambert_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_beer_lambert_mpi  # inputs
    "analysis_beer_lambert.py --geometry 1d --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_radiation_failure_test(
    test_1d_radiation_transport_kinetic_thermal_cold_rejected
    1
    inputs_base_1d_radiation_transport_kinetic_thermal_cold_rejected
    "nondegenerate particles in every affected NGP cell")

add_radiation_failure_test(
    test_3d_radiation_transport_packet_budget_negative_rejected
    3
    "inputs_base_3d_radiation_transport_packet_sampling radiation_transport.particle_conversion_target_packet_count=-1"
    "requires a nonnegative target packet count")
add_radiation_failure_test(
    test_3d_radiation_transport_packet_batch_negative_rejected
    3
    "inputs_base_3d_radiation_transport_group_sampling radiation_transport.particle_conversion_batch_size=-1"
    "batch size must be nonnegative")
add_radiation_failure_test(
    test_3d_radiation_transport_group_budget_size_rejected
    3
    "inputs_base_3d_radiation_transport_group_sampling radiation_transport.particle_conversion_group_target_packet_counts=256"
    "group packet budget count must match energy groups")
add_radiation_failure_test(
    test_3d_radiation_transport_group_budget_nonpositive_rejected
    3
    "inputs_base_3d_radiation_transport_group_sampling radiation_transport.particle_conversion_group_target_packet_counts='256 0'"
    "group packet budgets must be positive")
add_radiation_failure_test(
    test_3d_radiation_transport_group_budget_conflict_rejected
    3
    "inputs_base_3d_radiation_transport_group_sampling radiation_transport.particle_conversion_target_packet_count=256"
    "global and group packet budgets cannot be combined")
add_radiation_failure_test(
    test_3d_radiation_transport_packet_budget_zero_cap_rejected
    3
    "inputs_base_3d_radiation_transport_packet_sampling radiation_transport.particle_conversion_max_packets_per_cell=0"
    "requires a positive packet count cap")

add_radiation_failure_test(
    test_1d_radiation_transport_material_profile_negative_rejected
    1
    "inputs_base_1d_radiation_transport_material_profile_rejected hybrid_pic_model.initial_elec_temp(x,y,z)=-1.0"
    "positive electron temperature in eV")

add_radiation_failure_test(
    test_1d_radiation_transport_material_profile_static_closure_rejected
    1
    "inputs_base_1d_radiation_transport_material_profile_rejected hybrid_pic_model.solve_electron_energy_equation=0"
    "hybrid_pic_model.solve_electron_energy_equation=1.")

if(TEST test_1d_radiation_transport_opposing_group_work.analysis)
    add_test(
        NAME test_1d_radiation_transport_spectral_carry_fixture
        COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/prepare_spectral_carry_fixture.py
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_1d_radiation_transport_opposing_group_work/diags/chk000001)
    set_property(TEST test_1d_radiation_transport_spectral_carry_fixture
        PROPERTY DEPENDS test_1d_radiation_transport_opposing_group_work.analysis)
    add_radiation_failure_test(
        test_1d_radiation_transport_missing_spectral_carry_rejected
        1
        "inputs_base_1d_radiation_transport_spectral_carry_rejected amr.restart='../test_1d_radiation_transport_opposing_group_work/diags/chk000001_missing_spectral_carry'"
        "Checkpoint is missing the required MultiFab header")
    set_property(TEST test_1d_radiation_transport_missing_spectral_carry_rejected.run
        PROPERTY DEPENDS test_1d_radiation_transport_spectral_carry_fixture)
    if(TEST test_1d_radiation_transport_opposing_group_work.cleanup)
        set_property(TEST test_1d_radiation_transport_opposing_group_work.cleanup
            APPEND PROPERTY DEPENDS test_1d_radiation_transport_missing_spectral_carry_rejected.run)
    endif()
endif()

add_warpx_test(
    test_1d_radiation_transport_kinetic_thermal_drift  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_kinetic_thermal_drift  # inputs
    "analysis_kinetic_thermal_drift.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (analytic energy and momentum invariants are checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_kinetic_thermal_drift_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_kinetic_thermal_drift_mpi  # inputs
    "analysis_kinetic_thermal_drift.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (analytic energy and momentum invariants are checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_layered_interface  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_layered_interface  # inputs
    "analysis_layered_interface.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (piecewise Beer-Lambert attenuation is checked analytically)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_matter_exchange  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_matter_exchange  # inputs
    "analysis_matter_radiation_exchange.py --geometry 1d"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_radiation_transport_beer_lambert  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_radiation_transport_beer_lambert  # inputs
    "analysis_beer_lambert.py --geometry 2d --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_radiation_transport_matter_exchange  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_radiation_transport_matter_exchange  # inputs
    "analysis_matter_radiation_exchange.py --geometry 2d"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_beer_lambert  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_beer_lambert  # inputs
    "analysis_beer_lambert.py --geometry rcyl --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_electron_thermodynamics  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_electron_thermodynamics  # inputs
    "analysis_electron_thermodynamics_exchange.py --precision ${WarpX_PRECISION}"  # analysis
    OFF  # checksum (analytic EOS exchange and conservative ledgers are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_lte_hybrid_implicit_boundary  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_lte_hybrid_implicit_boundary  # inputs
    "analysis_lte_implicit_metric_boundary.py --geometry rcylinder --precision ${WarpX_PRECISION} --alpha-planck 1.0e5"  # analysis
    OFF  # checksum (analytic cylindrical corner-capacity metric and ledgers are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_lte_hybrid_implicit_boundary_mpi  # name
    RCYLINDER  # dims
    2  # nprocs
    inputs_test_rcylinder_radiation_transport_lte_hybrid_implicit_boundary_mpi  # inputs
    "analysis_lte_implicit_metric_boundary.py --geometry rcylinder --precision ${WarpX_PRECISION} --alpha-planck 1.0e5"  # analysis
    OFF  # checksum (analytic cylindrical corner-capacity metric and ledgers are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_lte_hybrid_implicit_boundary_control  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_lte_hybrid_implicit_boundary_control  # inputs
    "analysis_lte_implicit_metric_boundary.py --geometry rcylinder --precision ${WarpX_PRECISION} --alpha-planck 0.0"  # analysis
    OFF  # checksum (zero-opacity invariance is checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_lte_hybrid_implicit_boundary_control_mpi  # name
    RCYLINDER  # dims
    2  # nprocs
    inputs_test_rcylinder_radiation_transport_lte_hybrid_implicit_boundary_control_mpi  # inputs
    "analysis_lte_implicit_metric_boundary.py --geometry rcylinder --precision ${WarpX_PRECISION} --alpha-planck 0.0"  # analysis
    OFF  # checksum (zero-opacity invariance is checked)
    OFF  # dependency
)

add_warpx_test(
    test_rz_radiation_transport_hybrid_near_floor  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_radiation_transport_hybrid_near_floor  # inputs
    "analysis_hybrid_rz_near_floor_realization.py --precision ${WarpX_PRECISION}"  # analysis
    OFF  # checksum (metric realization and residual closure are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rz_radiation_transport_hybrid_near_floor_mpi  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_radiation_transport_hybrid_near_floor_mpi  # inputs
    "analysis_hybrid_rz_near_floor_realization.py --precision ${WarpX_PRECISION} --compare-reference"  # analysis
    OFF  # checksum (metric realization and serial equivalence are checked)
    test_rz_radiation_transport_hybrid_near_floor  # dependency
)

if(RCYLINDER IN_LIST WarpX_DIMS AND WarpX_APP
   AND WarpX_PRECISION STREQUAL "DOUBLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_warpx_test(
        test_rcylinder_radiation_transport_kinetic_cold_ring  # name
        RCYLINDER  # dims
        1  # nprocs
        inputs_test_rcylinder_radiation_transport_kinetic_cold_ring  # inputs
        "analysis_kinetic_cold_ring.py"  # analysis
        OFF  # checksum (local cylindrical moments and energy are checked)
        OFF  # dependency
    )

    add_warpx_test(
        test_rcylinder_radiation_transport_kinetic_cold_ring_mpi  # name
        RCYLINDER  # dims
        2  # nprocs
        inputs_test_rcylinder_radiation_transport_kinetic_cold_ring_mpi  # inputs
        "analysis_kinetic_cold_ring.py"  # analysis
        OFF  # checksum (local cylindrical moments and energy are checked)
        OFF  # dependency
    )
endif()

add_warpx_test(
    test_rcylinder_radiation_transport_latent_electron_thermodynamics  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_latent_electron_thermodynamics  # inputs
    "analysis_latent_electron_thermodynamics_exchange.py --precision ${WarpX_PRECISION}"  # analysis
    OFF  # checksum (nonlinear EOS exchange and conservative ledgers are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_matter_exchange  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_matter_exchange  # inputs
    "analysis_matter_radiation_exchange.py --geometry rcyl"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_boundary  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_boundary  # inputs
    "analysis_boundary.py --mode serial --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_boundary_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_boundary_mpi  # inputs
    "analysis_boundary.py --mode mpi --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_cartesian_cavity  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_cartesian_cavity  # inputs
    "analysis_diffusion_boundary.py --mode cavity --momentum-component 2 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_diffusion_boundary_reflecting  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_diffusion_boundary_reflecting  # inputs
    "analysis_diffusion_boundary.py --mode reflecting --momentum-component 2 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_diffusion_boundary_vacuum  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_diffusion_boundary_vacuum  # inputs
    "analysis_diffusion_boundary.py --mode vacuum --momentum-component 2 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_diffusion_boundary_restart  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_diffusion_boundary_restart  # inputs
    "analysis_boundary_restart.py --precision ${WarpX_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_diffusion_boundary_restart_restart  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_diffusion_boundary_restart_restart amr.restart='../test_1d_radiation_transport_diffusion_boundary_restart/diags/chk000006'"  # inputs
    "analysis_boundary_restart.py --compare-reference --precision ${WarpX_PRECISION}"  # analysis
    OFF  # checksum
    test_1d_radiation_transport_diffusion_boundary_restart  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_diffusion_boundary_marshak  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_diffusion_boundary_marshak  # inputs
    "analysis_diffusion_boundary.py --mode marshak --momentum-component 2 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

if(WarpX_PRECISION STREQUAL "DOUBLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_warpx_test(
        test_rcylinder_radiation_transport_absorption_momentum  # name
        RCYLINDER  # dims
        1  # nprocs
        inputs_test_rcylinder_radiation_transport_absorption_momentum  # inputs
        "analysis_momentum.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rcylinder_radiation_transport_absorption_moving_momentum  # name
        RCYLINDER  # dims
        1  # nprocs
        inputs_test_rcylinder_radiation_transport_absorption_moving_momentum  # inputs
        "analysis_moving_momentum.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

add_warpx_test(
    test_rcylinder_radiation_transport_diffusion_boundary_reflecting  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_diffusion_boundary_reflecting  # inputs
    "analysis_diffusion_boundary.py --mode reflecting --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_diffusion_boundary_vacuum  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_diffusion_boundary_vacuum  # inputs
    "analysis_diffusion_boundary.py --mode vacuum --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_diffusion_boundary_marshak  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_diffusion_boundary_marshak  # inputs
    "analysis_diffusion_boundary.py --mode marshak --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_streaming_boundary  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_streaming_boundary  # inputs
    "analysis_streaming_boundary_rcylinder.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_streaming_boundary_mpi  # name
    RCYLINDER  # dims
    2  # nprocs
    inputs_test_rcylinder_radiation_transport_streaming_boundary_mpi  # inputs
    "analysis_streaming_boundary_rcylinder.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_exact_layered  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_exact_layered  # inputs
    "analysis_exact_layered_rcylinder.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (piecewise circular-face attenuation is checked analytically)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_exact_layered_mpi  # name
    RCYLINDER  # dims
    2  # nprocs
    inputs_test_rcylinder_radiation_transport_exact_layered_mpi  # inputs
    "analysis_exact_layered_rcylinder.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (piecewise circular-face attenuation is checked analytically)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_exact_omp_tiling  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_exact_omp_tiling  # inputs
    "analysis_exact_omp_tiling_rcylinder.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (host material scatter is checked analytically)
    OFF  # dependency
)
if(TEST test_rcylinder_radiation_transport_exact_omp_tiling.run)
    set_property(
        TEST test_rcylinder_radiation_transport_exact_omp_tiling.run PROPERTY
        ENVIRONMENT
            "AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/"
            "OMP_NUM_THREADS=4")
endif()

add_warpx_test(
    test_rcylinder_radiation_transport_exact_axis  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_exact_axis  # inputs
    "analysis_exact_axis_rcylinder.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (axis traversal and the post-axis radial crossing are analytic)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_exact_outer  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_exact_outer  # inputs
    "analysis_exact_outer_rcylinder.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (outer-face attenuation and streaming loss are checked)
    OFF  # dependency
)

add_radiation_failure_test(
    test_rcylinder_radiation_transport_exact_annular_rejected
    RCYLINDER
    "inputs_test_rcylinder_radiation_transport_exact_layered geometry.prob_lo=0.1"
    "requires an axis-containing RCYLINDER domain")

add_radiation_failure_test(
    test_rcylinder_radiation_transport_exact_momentum_rejected
    RCYLINDER
    "inputs_test_rcylinder_radiation_transport_exact_layered radiation_transport.enable_momentum_coupling=1 radiation_transport.momentum_species=ions"
    "incompatible with radiation_transport.enable_momentum_coupling=1.")

add_radiation_failure_test(
    test_rcylinder_radiation_transport_exact_conversion_rejected
    RCYLINDER
    "inputs_test_rcylinder_radiation_transport_exact_layered radiation_transport.enable_particle_conversion=1 radiation_transport.emission_photon_energy=1.0e-19"
    "enable_particle_conversion=1.")

add_radiation_failure_test(
    test_rcylinder_radiation_transport_exact_boundary_rejected
    RCYLINDER
    "inputs_test_rcylinder_radiation_transport_exact_layered boundary.particle_hi=absorbing"
    "requires an Open outer radial particle boundary.")

add_radiation_failure_test(
    test_rz_radiation_transport_exact_rejected
    RZ
    inputs_base_rz_radiation_transport_exact_rejected
    "not supported in RZ geometry.")

add_radiation_failure_test(
    test_rsphere_radiation_transport_exact_rejected
    RSPHERE
    inputs_base_rsphere_radiation_transport_exact_rejected
    "not supported in RSPHERE geometry.")

if(WarpX_PRECISION STREQUAL "DOUBLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_warpx_test(
        test_rcylinder_radiation_transport_diffusion_momentum  # name
        RCYLINDER  # dims
        1  # nprocs
        inputs_test_rcylinder_radiation_transport_diffusion_momentum  # inputs
        "analysis_diffusion_momentum.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rcylinder_radiation_transport_diffusion_multigroup_momentum  # name
        RCYLINDER  # dims
        1  # nprocs
        inputs_test_rcylinder_radiation_transport_diffusion_multigroup_momentum  # inputs
        "analysis_diffusion_multigroup_momentum.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

add_warpx_test(
    test_rcylinder_radiation_transport_species_opacity  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_species_opacity  # inputs
    "analysis_species_opacity.py"  # analysis
    OFF  # checksum (species support and energy invariants are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_species_opacity_planck  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_species_opacity_planck  # inputs
    "analysis_species_opacity_planck.py"  # analysis
    OFF  # checksum (analytic Planck emission and invariants are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_species_opacity_packet  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_species_opacity_packet  # inputs
    "analysis_species_opacity_packet.py"  # analysis
    OFF  # checksum (analytic attenuation and energy invariant are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_species_opacity_conversion  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_radiation_transport_species_opacity_conversion  # inputs
    "analysis_species_opacity_conversion.py"  # analysis
    OFF  # checksum (additive Rosseland conversion and invariants are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_species_opacity_table_packet  # name
    RCYLINDER  # dims
    1  # nprocs
    "inputs_test_rcylinder_radiation_transport_species_opacity_table_packet radiation_transport.absorption_coefficient_table_file_foam='${CMAKE_CURRENT_SOURCE_DIR}/opacity_species_absorption_foam_3d_linear.dat'"  # inputs
    "analysis_species_opacity_table_packet.py"  # analysis
    OFF  # checksum (mixed backends, additive attenuation and invariants are checked)
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_radiation_transport_species_opacity_table_lte_diffusion  # name
    RCYLINDER  # dims
    1  # nprocs
    "inputs_test_rcylinder_radiation_transport_species_opacity_table_lte_diffusion radiation_transport.planck_absorption_coefficient_table_file_foam='${CMAKE_CURRENT_SOURCE_DIR}/opacity_species_planck_foam_2d_linear.dat' radiation_transport.planck_absorption_coefficient_spectral_table_file_tungsten='${CMAKE_CURRENT_SOURCE_DIR}/opacity_species_planck_tungsten_3d_linear.dat' radiation_transport.rosseland_transport_coefficient_spectral_table_file_foam='${CMAKE_CURRENT_SOURCE_DIR}/opacity_species_rosseland_foam_3d_linear.dat' radiation_transport.rosseland_transport_coefficient_table_file_tungsten='${CMAKE_CURRENT_SOURCE_DIR}/opacity_species_rosseland_tungsten_2d_linear.dat'"  # inputs
    "analysis_species_opacity_table_lte_diffusion.py"  # analysis
    OFF  # checksum (mixed table backends, cylindrical FLD and invariants are checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_conversion  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_conversion  # inputs
    "analysis_lte.py --coupling hybrid --diffusion --conversion --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_radiation_failure_test(
    test_1d_radiation_transport_conversion_with_momentum_rejected
    1
    inputs_base_1d_radiation_transport_conversion_with_momentum_rejected
    "does not retain the converted packet momentum")

add_warpx_test(
    test_1d_radiation_transport_conversion_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_conversion_mpi  # inputs
    "analysis_lte.py --coupling hybrid --diffusion --conversion --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_diffusion  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_diffusion  # inputs
    "analysis_lte.py --coupling hybrid --diffusion --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

if(WarpX_PRECISION STREQUAL "DOUBLE" AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_warpx_test(
        test_1d_radiation_transport_scattering_recoil  # name
        1  # dims
        1  # nprocs
        "inputs_base_1d_radiation_transport_scattering_recoil my_constants.alpha_control=0.0"  # inputs
        "analysis_scattering_recoil.py --planck-absorption 0.0 --precision ${WarpX_PRECISION}"  # analysis
        OFF  # checksum (analytic FLD, recoil and energy ledgers are checked)
        OFF  # dependency
    )

    add_warpx_test(
        test_1d_radiation_transport_scattering_recoil_absorption_control  # name
        1  # dims
        1  # nprocs
        "inputs_base_1d_radiation_transport_scattering_recoil my_constants.alpha_control=2.0"  # inputs
        "analysis_scattering_recoil.py --planck-absorption 2.0 --precision ${WarpX_PRECISION} --reference ../test_1d_radiation_transport_scattering_recoil"  # analysis
        OFF  # checksum (analytic FLD, recoil and energy ledgers are checked)
        test_1d_radiation_transport_scattering_recoil  # dependency
    )

    add_warpx_test(
        test_1d_radiation_transport_momentum_carry  # name
        1  # dims
        1  # nprocs
        inputs_test_1d_radiation_transport_momentum_carry  # inputs
        "analysis_momentum_carry.py"  # analysis
        OFF  # checksum (sub-ULP impulse, applied work, and carry are checked)
        OFF  # dependency
    )

    add_warpx_test(
        test_1d_radiation_transport_streaming_momentum_carry  # name
        1  # dims
        1  # nprocs
        inputs_base_1d_radiation_transport_streaming_momentum_carry  # inputs
        "analysis_streaming_momentum_carry.py"  # analysis
        OFF  # checksum (sub-ULP streaming recoil and signed work are checked)
        OFF  # dependency
    )

    add_warpx_test(
        test_1d_radiation_particle_carry
        1
        1
        inputs_base_1d_radiation_particle_carry
        "analysis_particle_carry.py"
        OFF
        OFF
    )
    add_warpx_test(
        test_1d_radiation_particle_carry_restart
        1
        2
        "inputs_base_1d_radiation_particle_carry amr.restart='../test_1d_radiation_particle_carry/diags/chk000015'"
        "analysis_particle_carry.py --reference ../test_1d_radiation_particle_carry"
        OFF
        test_1d_radiation_particle_carry
    )
    add_warpx_test(
        test_2d_radiation_particle_carry
        2
        1
        inputs_base_2d_radiation_particle_carry
        "analysis_particle_carry.py"
        OFF
        OFF
    )
    add_warpx_test(
        test_2d_radiation_particle_carry_restart
        2
        2
        "inputs_base_2d_radiation_particle_carry amr.restart='../test_2d_radiation_particle_carry/diags/chk000015'"
        "analysis_particle_carry.py --reference ../test_2d_radiation_particle_carry"
        OFF
        test_2d_radiation_particle_carry
    )
    add_radiation_failure_test(
        test_1d_radiation_particle_carry_boundary_rejected 1
        "inputs_base_1d_radiation_particle_carry boundary.field_lo=pec boundary.field_hi=pec boundary.particle_lo=absorbing boundary.particle_hi=absorbing"
        "initially requires periodic")
    add_radiation_failure_test(
        test_1d_radiation_particle_carry_ownership_rejected 1
        "inputs_base_1d_radiation_particle_carry radiation_transport.momentum_carry=cell amr.restart='../test_1d_radiation_particle_carry/diags/chk000015'"
        "preserve radiation momentum_carry ownership")
    if(TEST test_1d_radiation_particle_carry_ownership_rejected.run)
        set_property(TEST test_1d_radiation_particle_carry_ownership_rejected.run
            APPEND PROPERTY DEPENDS test_1d_radiation_particle_carry.analysis)
        if(TEST test_1d_radiation_particle_carry.cleanup)
            set_property(TEST test_1d_radiation_particle_carry.cleanup
                APPEND PROPERTY DEPENDS
                    test_1d_radiation_particle_carry_ownership_rejected.run)
        endif()
    endif()
    foreach(carry_restart IN ITEMS
            test_1d_radiation_particle_carry_restart
            test_2d_radiation_particle_carry_restart
            test_1d_radiation_transport_moving_beam_restart)
        if(TEST ${carry_restart}.run)
            add_test(NAME ${carry_restart}_prepare COMMAND ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/prepare_particle_carry_restart.py
                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${carry_restart})
            set_tests_properties(${carry_restart}_prepare PROPERTIES
                FIXTURES_SETUP ${carry_restart}_fresh_output)
            set_tests_properties(${carry_restart}.run PROPERTIES
                FIXTURES_REQUIRED ${carry_restart}_fresh_output)
        endif()
    endforeach()

    add_warpx_test(
        test_1d_radiation_transport_streaming_momentum_carry_lte_frozen  # name
        1  # dims
        1  # nprocs
        "inputs_base_1d_radiation_transport_streaming_momentum_carry radiation_transport.enable_lte_exchange=1 radiation_transport.planck_absorption_coefficient=0.0"  # inputs
        "analysis_streaming_momentum_carry.py"  # analysis
        OFF  # checksum (signed carry work passes through frozen LTE safely)
        OFF  # dependency
    )

    add_warpx_test(
        test_1d_radiation_transport_streaming_momentum_carry_lte_implicit  # name
        1  # dims
        1  # nprocs
        "inputs_base_1d_radiation_transport_streaming_momentum_carry radiation_transport.enable_lte_exchange=1 radiation_transport.planck_absorption_coefficient=0.0 radiation_transport.lte_exchange_solver=implicit_temperature"  # inputs
        "analysis_streaming_momentum_carry.py"  # analysis
        OFF  # checksum (signed carry work is reserved before implicit LTE)
        OFF  # dependency
    )

    add_warpx_test(
        test_1d_radiation_transport_streaming_momentum_carry_lte_nonlinear  # name
        1  # dims
        1  # nprocs
        "inputs_base_1d_radiation_transport_streaming_momentum_carry radiation_transport.enable_lte_exchange=1 radiation_transport.planck_absorption_coefficient=0.0 radiation_transport.lte_exchange_solver=implicit_temperature hybrid_pic_model.electron_thermodynamics=fixed_charge_latent_energy hybrid_pic_model.electron_latent_transition_temperature_eV=1.0e20 hybrid_pic_model.electron_latent_energy_eV=1.0e19 hybrid_pic_model.electron_latent_sharpness=4.0"  # inputs
        "analysis_streaming_momentum_carry.py"  # analysis
        OFF  # checksum (signed carry work passes through nonlinear LTE remap)
        OFF  # dependency
    )

    add_warpx_test(
        test_1d_radiation_transport_streaming_momentum_carry_lte_emission
        1
        1
        inputs_test_1d_radiation_transport_streaming_momentum_carry_lte_emission
        "analysis_streaming_momentum_carry.py --lte-emission"
        OFF
        OFF
    )

    add_radiation_failure_test(
        test_1d_radiation_transport_streaming_momentum_moving_window_rejected
        1
        "inputs_base_1d_radiation_transport_streaming_momentum_carry boundary.field_lo=pec boundary.field_hi=pec boundary.particle_lo=absorbing boundary.particle_hi=absorbing warpx.do_moving_window=1 warpx.moving_window_dir=z warpx.moving_window_v=1.0"
        "Radiation momentum coupling is not supported with a moving window")

    add_warpx_test(
        test_1d_radiation_transport_momentum_carry_restart  # name
        1  # dims
        1  # nprocs
        "inputs_test_1d_radiation_transport_momentum_carry_restart amr.restart='../test_1d_radiation_transport_momentum_carry/diags/chk000149'"  # inputs
        "analysis_momentum_carry.py --compare-reference"  # analysis
        OFF  # checksum (nonzero checkpointed carry and restart are checked)
        test_1d_radiation_transport_momentum_carry  # dependency
    )

    if(TEST test_1d_radiation_transport_momentum_carry.analysis)
        set(momentum_carry_fixture_dir
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_1d_radiation_transport_momentum_carry/diags")
        add_test(
            NAME test_1d_radiation_transport_momentum_carry_restart_fixtures
            COMMAND
                ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/prepare_momentum_carry_restart_fixtures.py
                ${momentum_carry_fixture_dir}/chk000149
                ${momentum_carry_fixture_dir}/chk000149_legacy_missing_carry
                ${momentum_carry_fixture_dir}/chk000149_corrupt_missing_carry)
        set_property(
            TEST test_1d_radiation_transport_momentum_carry_restart_fixtures
            APPEND PROPERTY DEPENDS
                test_1d_radiation_transport_momentum_carry.analysis)

        add_warpx_test(
            test_1d_radiation_transport_momentum_carry_legacy_restart  # name
            1  # dims
            1  # nprocs
            "inputs_base_1d_radiation_transport_momentum_carry_legacy_restart amr.restart='../test_1d_radiation_transport_momentum_carry/diags/chk000149_legacy_missing_carry'"  # inputs
            "analysis_momentum_carry_legacy_restart.py"  # analysis
            OFF  # checksum (the immediate restart diagnostic checks zero carry)
            test_1d_radiation_transport_momentum_carry  # dependency
        )
        if(TEST test_1d_radiation_transport_momentum_carry_legacy_restart.run)
            set_property(
                TEST test_1d_radiation_transport_momentum_carry_legacy_restart.run
                APPEND PROPERTY DEPENDS
                    test_1d_radiation_transport_momentum_carry_restart_fixtures)
        endif()

        add_warpx_test(
            test_1d_radiation_transport_momentum_carry_legacy_mpi_restart  # name
            1  # dims
            2  # nprocs
            "inputs_base_1d_radiation_transport_momentum_carry_legacy_restart amr.restart='../test_1d_radiation_transport_momentum_carry/diags/chk000149_legacy_missing_carry'"  # inputs
            "analysis_momentum_carry_legacy_restart.py"  # analysis
            OFF  # checksum (MPI ranks must take the same optional-field branch)
            test_1d_radiation_transport_momentum_carry  # dependency
        )
        if(TEST test_1d_radiation_transport_momentum_carry_legacy_mpi_restart.run)
            set_property(
                TEST test_1d_radiation_transport_momentum_carry_legacy_mpi_restart.run
                APPEND PROPERTY DEPENDS
                    test_1d_radiation_transport_momentum_carry_restart_fixtures)
        endif()

        add_radiation_failure_test(
            test_1d_radiation_transport_momentum_carry_corrupt_restart
            1
            "inputs_base_1d_radiation_transport_momentum_carry_corrupt_restart amr.restart='../test_1d_radiation_transport_momentum_carry/diags/chk000149_corrupt_missing_carry'"
            "missing the required MultiFab header")
        if(TEST test_1d_radiation_transport_momentum_carry_corrupt_restart.run)
            set_property(
                TEST test_1d_radiation_transport_momentum_carry_corrupt_restart.run
                APPEND PROPERTY DEPENDS
                    test_1d_radiation_transport_momentum_carry_restart_fixtures)
        endif()

        add_radiation_failure_test(
            test_1d_radiation_transport_momentum_carry_corrupt_restart_mpi
            1
            "inputs_base_1d_radiation_transport_momentum_carry_corrupt_restart amr.restart='../test_1d_radiation_transport_momentum_carry/diags/chk000149_corrupt_missing_carry'"
            "missing the required MultiFab header"
            2)
        if(TEST test_1d_radiation_transport_momentum_carry_corrupt_restart_mpi.run)
            set_property(
                TEST test_1d_radiation_transport_momentum_carry_corrupt_restart_mpi.run
                APPEND PROPERTY DEPENDS
                    test_1d_radiation_transport_momentum_carry_restart_fixtures)
        endif()

        add_radiation_failure_test(
            test_1d_radiation_transport_momentum_carry_disabled_restart
            1
            "inputs_base_1d_radiation_transport_momentum_carry_corrupt_restart amr.restart='../test_1d_radiation_transport_momentum_carry/diags/chk000149' radiation_transport.enable_momentum_coupling=0"
            "checkpoint contains radiation momentum-carry fields")
        if(TEST test_1d_radiation_transport_momentum_carry_disabled_restart.run)
            set_property(
                TEST test_1d_radiation_transport_momentum_carry_disabled_restart.run
                APPEND PROPERTY DEPENDS
                    test_1d_radiation_transport_momentum_carry.analysis)
        endif()

        if(WarpX_TEST_CLEANUP
           AND TEST test_1d_radiation_transport_momentum_carry.cleanup)
            foreach(corrupt_test
                    test_1d_radiation_transport_momentum_carry_corrupt_restart
                    test_1d_radiation_transport_momentum_carry_corrupt_restart_mpi
                    test_1d_radiation_transport_momentum_carry_disabled_restart)
                if(TEST ${corrupt_test}.run)
                    set_property(
                        TEST test_1d_radiation_transport_momentum_carry.cleanup
                        APPEND PROPERTY DEPENDS ${corrupt_test}.run)
                endif()
            endforeach()
        endif()
    endif()
endif()

add_warpx_test(
    test_1d_radiation_transport_cold_start_profiles  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_cold_start_profiles  # inputs
    "analysis_cold_start_profiles.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (analytic radiation cold-start profile is checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_diffusion_only  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_diffusion_only  # inputs
    "analysis_diffusion_only.py"  # analysis
    OFF  # checksum (analytic conservation and stencil symmetry are checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_diffusion_read_only_hybrid  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_diffusion_read_only_hybrid  # inputs
    "analysis_diffusion_only.py"  # analysis
    OFF  # checksum (Hybrid-PIC supplies opacity state without LTE write-back)
    OFF  # dependency
)

add_radiation_failure_test(
    test_1d_radiation_transport_diffusion_only_table_rejected
    1
    "inputs_base_1d_radiation_transport_diffusion_only_table_rejected radiation_transport.rosseland_transport_coefficient_table_file=${CMAKE_CURRENT_SOURCE_DIR}/opacity_rosseland_2d_log.dat"
    "State-dependent Rosseland tables and per-species opacity require")

add_radiation_failure_test(
    test_1d_radiation_transport_diffusion_moving_window_rejected
    1
    "inputs_test_1d_radiation_transport_diffusion_only boundary.field_lo=pec boundary.field_hi=pec boundary.particle_lo=absorbing boundary.particle_hi=absorbing warpx.do_moving_window=1 warpx.moving_window_dir=z warpx.moving_window_v=1.0"
    "Radiation LTE/diffusion transport is not supported")

add_warpx_test(
    test_1d_radiation_transport_cold_start_profiles_multigroup  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_cold_start_profiles_multigroup  # inputs
    "analysis_cold_start_profiles.py --multigroup --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (radiation group initialization and zero defaults are checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_cold_start_profiles_restart  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_cold_start_profiles_restart amr.restart='../test_1d_radiation_transport_cold_start_profiles/diags/chk000001'"  # inputs
    "analysis_cold_start_profiles.py --restart --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (checkpointed radiation energy is checked)
    test_1d_radiation_transport_cold_start_profiles  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_hybrid  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_hybrid  # inputs
    "analysis.py diags/diag1000001 --coupling hybrid --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_kinetic  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_kinetic  # inputs
    "analysis.py diags/diag1000001 --coupling kinetic --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_kinetic_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_kinetic_mpi  # inputs
    "analysis.py diags/diag1000001 --coupling kinetic --require-multibox-residual --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_kinetic_zero_opacity_control  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_kinetic_zero_opacity_control  # inputs
    "analysis_kinetic_zero_opacity.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_lte_hybrid  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_lte_hybrid  # inputs
    "analysis_lte.py --coupling hybrid --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_lte_hybrid_implicit  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_lte_hybrid_implicit  # inputs
    "analysis_lte_implicit.py --coupling hybrid --multigroup --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_lte_hybrid_implicit_boundary  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_lte_hybrid_implicit_boundary  # inputs
    "analysis_lte_implicit_boundary.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --alpha-planck 1.0e5"  # analysis
    OFF  # checksum (analytic boundary capacity and ledgers are checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_lte_hybrid_implicit_boundary_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_lte_hybrid_implicit_boundary_mpi  # inputs
    "analysis_lte_implicit_boundary.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --alpha-planck 1.0e5"  # analysis
    OFF  # checksum (analytic boundary capacity and ledgers are checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_lte_hybrid_implicit_boundary_control  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_lte_hybrid_implicit_boundary_control  # inputs
    "analysis_lte_implicit_boundary.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --alpha-planck 0.0"  # analysis
    OFF  # checksum (zero-opacity invariance is checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_lte_hybrid_implicit_boundary_control_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_lte_hybrid_implicit_boundary_control_mpi  # inputs
    "analysis_lte_implicit_boundary.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --alpha-planck 0.0"  # analysis
    OFF  # checksum (zero-opacity invariance is checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_hybrid_near_floor  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_hybrid_near_floor  # inputs
    "analysis_hybrid_near_floor_realization.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (below-ULP EOS realization and residual are checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_hybrid_near_floor_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_hybrid_near_floor_mpi  # inputs
    "analysis_hybrid_near_floor_realization.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --compare-reference"  # analysis
    OFF  # checksum (below-ULP EOS realization and serial equivalence are checked)
    test_1d_radiation_transport_hybrid_near_floor  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_lte_kinetic  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_lte_kinetic  # inputs
    "analysis_lte.py --coupling kinetic --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_lte_kinetic_implicit  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_lte_kinetic_implicit  # inputs
    "analysis_lte_implicit.py --coupling kinetic --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_multigroup  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_multigroup  # inputs
    "analysis_multigroup.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_multigroup_kinetic  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_multigroup_kinetic  # inputs
    "analysis_multigroup_kinetic.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_multigroup_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_multigroup_mpi  # inputs
    "analysis_multigroup.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_multigroup_opacity_table  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_multigroup_opacity_table radiation_transport.planck_absorption_coefficient_spectral_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_planck_multigroup_3d_linear.dat' radiation_transport.rosseland_transport_coefficient_spectral_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_rosseland_multigroup_3d_linear.dat'"  # inputs
    "analysis_multigroup.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_multigroup_restart  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_multigroup_restart  # inputs
    "analysis_restart.py diags/diag1000002"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_multigroup_restart_restart  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_multigroup_restart_restart amr.restart='../test_1d_radiation_transport_multigroup_restart/diags/chk000001'"  # inputs
    "analysis_restart.py diags/diag1000002 --compare-reference"  # analysis
    OFF  # checksum
    test_1d_radiation_transport_multigroup_restart  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_opacity_table  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_opacity_table radiation_transport.absorption_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_absorption_3d_log.dat' radiation_transport.planck_absorption_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_planck_2d_log.dat' radiation_transport.rosseland_transport_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_rosseland_2d_log.dat'"  # inputs
    "analysis_opacity_table.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_opacity_table_clamp  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_opacity_table_clamp my_constants.n0=1.0e22 radiation_transport.absorption_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_absorption_3d_log.dat' radiation_transport.planck_absorption_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_planck_2d_log.dat' radiation_transport.rosseland_transport_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_rosseland_2d_log.dat'"  # inputs
    "analysis_opacity_table.py --alpha 100.0 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_opacity_table_kinetic  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_opacity_table_kinetic radiation_transport.absorption_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_absorption_3d_log.dat'"  # inputs
    "analysis.py diags/diag1000001 --coupling kinetic --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_opacity_table_linear  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_opacity_table_linear radiation_transport.opacity_table_interpolation=linear radiation_transport.absorption_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_absorption_3d_log.dat' radiation_transport.planck_absorption_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_planck_2d_log.dat' radiation_transport.rosseland_transport_coefficient_table_file='${CMAKE_CURRENT_SOURCE_DIR}/opacity_rosseland_2d_log.dat'"  # inputs
    "analysis_opacity_table.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

if(WarpX_MATERIAL_OPACITY_HDF5 AND 1 IN_LIST WarpX_DIMS AND WarpX_APP)
    set(registered_opacity_fixture_dir
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/material_opacity_runtime_fixtures")
    set(registered_opacity_fixture_args
        "materials.material_a.opacity_table_file='${registered_opacity_fixture_dir}/material_a.h5' materials.material_b.opacity_table_file='${registered_opacity_fixture_dir}/material_b.h5'")
    set(registered_lte_opacity_fixture_args
        "materials.material_a.opacity_table_file='${registered_opacity_fixture_dir}/material_lte_a.h5' materials.material_b.opacity_table_file='${registered_opacity_fixture_dir}/material_lte_b.h5'")

    add_warpx_test(
        test_1d_radiation_transport_registered_material_opacity_order_a
        1
        1
        "inputs_base_1d_radiation_transport_registered_material_opacity_order_a ${registered_opacity_fixture_args}"
        "analysis_registered_material_opacity.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
        OFF
        OFF
    )
    set_tests_properties(
        test_1d_radiation_transport_registered_material_opacity_order_a.run
        PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)

    add_warpx_test(
        test_1d_radiation_transport_registered_material_opacity_order_a_mpi
        1
        2
        "inputs_base_1d_radiation_transport_registered_material_opacity_order_a ${registered_opacity_fixture_args}"
        "analysis_registered_material_opacity.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
        OFF
        OFF
    )
    if(TEST test_1d_radiation_transport_registered_material_opacity_order_a_mpi.run)
        set_tests_properties(
            test_1d_radiation_transport_registered_material_opacity_order_a_mpi.run
            PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)
    endif()

    add_warpx_test(
        test_1d_radiation_transport_registered_material_opacity_order_b
        1
        1
        "inputs_test_1d_radiation_transport_registered_material_opacity_order_b ${registered_opacity_fixture_args}"
        "analysis_registered_material_opacity.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --reference ../test_1d_radiation_transport_registered_material_opacity_order_a"
        OFF
        test_1d_radiation_transport_registered_material_opacity_order_a
    )
    set_tests_properties(
        test_1d_radiation_transport_registered_material_opacity_order_b.run
        PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)

    add_warpx_test(
        test_1d_radiation_transport_registered_material_opacity_kinetic
        1
        1
        "inputs_test_1d_radiation_transport_registered_material_opacity_kinetic ${registered_opacity_fixture_args}"
        "analysis_registered_material_opacity.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --kinetic"
        OFF
        OFF
    )
    set_tests_properties(
        test_1d_radiation_transport_registered_material_opacity_kinetic.run
        PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)

    add_warpx_test(
        test_1d_radiation_transport_registered_material_opacity_kinetic_mpi
        1
        2
        "inputs_test_1d_radiation_transport_registered_material_opacity_kinetic_mpi ${registered_opacity_fixture_args}"
        "analysis_registered_material_opacity.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION} --kinetic"
        OFF
        OFF
    )
    if(TEST test_1d_radiation_transport_registered_material_opacity_kinetic_mpi.run)
        set_tests_properties(
            test_1d_radiation_transport_registered_material_opacity_kinetic_mpi.run
            PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)
    endif()

    add_warpx_test(
        test_1d_radiation_transport_registered_material_opacity_legacy_pair
        1
        1
        "inputs_test_1d_radiation_transport_registered_material_opacity_legacy_pair materials.material_a.opacity_table_file='${registered_opacity_fixture_dir}/material_a.h5' radiation_transport.material_opacity_table_file='${registered_opacity_fixture_dir}/material_a.h5'"
        OFF
        OFF
        OFF
    )
    set_tests_properties(
        test_1d_radiation_transport_registered_material_opacity_legacy_pair.run
        PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)

    add_warpx_test(
        test_1d_radiation_transport_registered_material_opacity_lte
        1
        1
        "inputs_test_1d_radiation_transport_registered_material_opacity_lte ${registered_lte_opacity_fixture_args}"
        "analysis_registered_material_opacity_lte.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
        OFF
        OFF
    )
    set_tests_properties(
        test_1d_radiation_transport_registered_material_opacity_lte.run
        PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)

    add_warpx_test(
        test_1d_radiation_transport_registered_material_opacity_lte_mpi
        1
        2
        "inputs_test_1d_radiation_transport_registered_material_opacity_lte_mpi ${registered_lte_opacity_fixture_args}"
        "analysis_registered_material_opacity_lte.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"
        OFF
        OFF
    )
    if(TEST test_1d_radiation_transport_registered_material_opacity_lte_mpi.run)
        set_tests_properties(
            test_1d_radiation_transport_registered_material_opacity_lte_mpi.run
            PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)
    endif()

    foreach(failure_case IN ITEMS mixed group_mismatch key_mismatch)
        set(failure_test_name
            "test_1d_radiation_transport_registered_material_opacity_${failure_case}_rejected")
        set(failure_working_dir
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${failure_test_name}")
        file(MAKE_DIRECTORY "${failure_working_dir}")
        add_test(
            NAME "${failure_test_name}"
            COMMAND ${CMAKE_COMMAND}
                -DTEST_EXE=$<TARGET_FILE:app_1d>
                -DTEST_CASE=${failure_case}
                -DFIXTURE_DIR=${registered_opacity_fixture_dir}
                -DINPUT_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_registered_material_opacity_failure.cmake
            WORKING_DIRECTORY "${failure_working_dir}")
        set_tests_properties(
            "${failure_test_name}"
            PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)
    endforeach()
endif()

add_warpx_test(
    test_1d_radiation_transport_path  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_path  # inputs
    "analysis_path.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_path_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_path_mpi  # inputs
    "analysis_path.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_restart  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_restart  # inputs
    "analysis_restart.py diags/diag1000002"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_restart_restart  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_restart_restart amr.restart='../test_1d_radiation_transport_restart/diags/chk000001'"  # inputs
    "analysis_restart.py diags/diag1000002 --compare-reference"  # analysis
    OFF  # checksum
    test_1d_radiation_transport_restart  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_hybrid_realization_restart  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_hybrid_realization_restart  # inputs
    "analysis_hybrid_realization_restart.py --precision ${WarpX_PRECISION}"  # analysis
    OFF  # checksum (checkpointed material-realization residual is checked)
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_hybrid_realization_restart_restart  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_radiation_transport_hybrid_realization_restart_restart amr.restart='../test_1d_radiation_transport_hybrid_realization_restart/diags/chk000002'"  # inputs
    "analysis_hybrid_realization_restart.py --precision ${WarpX_PRECISION} --compare-reference"  # analysis
    OFF  # checksum (restart equivalence and cumulative residual are checked)
    test_1d_radiation_transport_hybrid_realization_restart  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_thermalization  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_radiation_transport_thermalization  # inputs
    "analysis_thermalization.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_radiation_transport_thermalization_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_radiation_transport_thermalization_mpi  # inputs
    "analysis_thermalization.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum
    OFF  # dependency
)

if(RZ IN_LIST WarpX_DIMS AND WarpX_APP)
    add_warpx_test(
        test_rz_radiation_transport_lte_hybrid_implicit_boundary  # name
        RZ  # dims
        1  # nprocs
        inputs_test_rz_radiation_transport_lte_hybrid_implicit_boundary  # inputs
        "analysis_lte_implicit_metric_boundary.py --geometry rz --precision ${WarpX_PRECISION} --alpha-planck 1.0e5"  # analysis
        OFF  # checksum (analytic cylindrical corner-capacity metric and ledgers are checked)
        OFF  # dependency
    )

    add_warpx_test(
        test_rz_radiation_transport_lte_hybrid_implicit_boundary_mpi  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_radiation_transport_lte_hybrid_implicit_boundary_mpi  # inputs
        "analysis_lte_implicit_metric_boundary.py --geometry rz --precision ${WarpX_PRECISION} --alpha-planck 1.0e5"  # analysis
        OFF  # checksum (analytic cylindrical corner-capacity metric and ledgers are checked)
        OFF  # dependency
    )

    add_warpx_test(
        test_rz_radiation_transport_lte_hybrid_implicit_boundary_control  # name
        RZ  # dims
        1  # nprocs
        inputs_test_rz_radiation_transport_lte_hybrid_implicit_boundary_control  # inputs
        "analysis_lte_implicit_metric_boundary.py --geometry rz --precision ${WarpX_PRECISION} --alpha-planck 0.0"  # analysis
        OFF  # checksum (zero-opacity invariance is checked)
        OFF  # dependency
    )

    add_warpx_test(
        test_rz_radiation_transport_lte_hybrid_implicit_boundary_control_mpi  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_radiation_transport_lte_hybrid_implicit_boundary_control_mpi  # inputs
        "analysis_lte_implicit_metric_boundary.py --geometry rz --precision ${WarpX_PRECISION} --alpha-planck 0.0"  # analysis
        OFF  # checksum (zero-opacity invariance is checked)
        OFF  # dependency
    )
endif()

if(WarpX_PRECISION STREQUAL "DOUBLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    if(RZ IN_LIST WarpX_DIMS AND WarpX_APP)
        add_warpx_test(
            test_rz_radiation_transport_kinetic_cold_ring  # name
            RZ  # dims
            1  # nprocs
            inputs_test_rz_radiation_transport_kinetic_cold_ring  # inputs
            "analysis_kinetic_cold_ring.py"  # analysis
            OFF  # checksum (local cylindrical moments and energy are checked)
            OFF  # dependency
        )
        add_warpx_test(
            test_rz_radiation_transport_kinetic_cold_ring_mpi  # name
            RZ  # dims
            2  # nprocs
            inputs_test_rz_radiation_transport_kinetic_cold_ring_mpi  # inputs
            "analysis_kinetic_cold_ring.py"  # analysis
            OFF  # checksum (local cylindrical moments and energy are checked)
            OFF  # dependency
        )
    endif()

    add_warpx_test(
        test_rz_radiation_transport_absorption_momentum  # name
        RZ  # dims
        1  # nprocs
        inputs_test_rz_radiation_transport_absorption_momentum  # inputs
        "analysis_momentum.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_3d_radiation_transport_absorption_momentum  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_radiation_transport_absorption_momentum  # inputs
        "analysis_momentum.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

add_radiation_failure_test(
    test_rsphere_radiation_transport_momentum_rejected
    RSPHERE
    inputs_base_rsphere_radiation_transport_momentum_rejected
    "RSPHERE until Cartesian-to-spherical particle initialization")

add_warpx_test(
    test_rz_radiation_transport_conversion  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_radiation_transport_conversion  # inputs
    "analysis_rz.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_radiation_transport_multigroup_conversion  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_radiation_transport_multigroup_conversion  # inputs
    "analysis_rz.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_radiation_transport_conversion  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_radiation_transport_conversion  # inputs
    "analysis_rz.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_radiation_transport_conversion_random_seed_checkpoint  # name
    3  # dims
    1  # nprocs
    inputs_base_3d_radiation_transport_conversion_random_seed_restart  # inputs
    "analysis_conversion_random_seed_restart.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (conversion timing and checkpoint production are checked)
    OFF  # dependency
)

add_warpx_test(
    test_3d_radiation_transport_conversion_random_seed_restart  # name
    3  # dims
    1  # nprocs
    "inputs_base_3d_radiation_transport_conversion_random_seed_restart_restart amr.restart='../test_3d_radiation_transport_conversion_random_seed_checkpoint/diags/chk000001'"  # inputs
    "analysis_conversion_random_seed_restart.py --reference ../test_3d_radiation_transport_conversion_random_seed_checkpoint --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (resolved random seed and packet realization are compared exactly)
    test_3d_radiation_transport_conversion_random_seed_checkpoint  # dependency
)

add_warpx_test(
    test_3d_radiation_transport_conversion_disabled_seed_checkpoint  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_radiation_transport_conversion_disabled_seed_checkpoint  # inputs
    OFF  # analysis (the dependent restart validates the saved seed)
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_radiation_transport_conversion_enabled_seed_restart  # name
    3  # dims
    1  # nprocs
    "inputs_base_3d_radiation_transport_conversion_random_seed_restart_restart amr.restart='../test_3d_radiation_transport_conversion_disabled_seed_checkpoint/diags/chk000001'"  # inputs
    "analysis_conversion_random_seed_restart.py --seed-checkpoint ../test_3d_radiation_transport_conversion_disabled_seed_checkpoint/diags/chk000001 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (conversion-disabled checkpoint seed is restored exactly)
    test_3d_radiation_transport_conversion_disabled_seed_checkpoint  # dependency
)

if(WarpX_EB)
    add_radiation_failure_test(
        test_3d_radiation_transport_bath_eb_rejected
        3
        "inputs_base_3d_radiation_transport_bath eb2.geom_type=all_regular"
        "marshak_bath is not supported with embedded boundaries")
    add_radiation_failure_test(
        test_3d_radiation_transport_conversion_eb_rejected
        3
        "inputs_base_3d_radiation_transport_packet_sampling eb2.geom_type=all_regular"
        "enable_particle_conversion=1 is not supported")
endif()

add_warpx_test(
    test_3d_radiation_transport_packet_sampling_seed42  # name
    3  # dims
    1  # nprocs
    inputs_base_3d_radiation_transport_packet_sampling  # inputs
    "analysis_packet_sampling.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (packet count, strata, and exact energy are checked)
    OFF  # dependency
)

add_warpx_test(
    test_3d_radiation_transport_packet_sampling_offset  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_radiation_transport_packet_sampling_offset  # inputs
    "analysis_packet_sampling.py --domain-offset 100000.1 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (mixed-precision offset cell ownership is checked)
    OFF  # dependency
)

add_warpx_test(
    test_3d_radiation_transport_packet_sampling_seed42_uniform  # name
    3  # dims
    1  # nprocs
    inputs_base_3d_radiation_transport_packet_sampling_uniform  # inputs
    "analysis_packet_sampling.py --uniform-domain --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (all 64 cells in the one-box serial reference are checked)
    OFF  # dependency
)

add_warpx_test(
    test_3d_radiation_transport_packet_sampling_seed42_mpi_multibox  # name
    3  # dims
    2  # nprocs
    inputs_base_3d_radiation_transport_packet_sampling_mpi  # inputs
    "analysis_packet_sampling.py --uniform-domain --reference ../test_3d_radiation_transport_packet_sampling_seed42_uniform --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (MPI/multibox packet records are compared exactly to serial)
    test_3d_radiation_transport_packet_sampling_seed42_uniform  # dependency
)

add_warpx_test(
    test_3d_radiation_transport_packet_sampling_seed42_repeat  # name
    3  # dims
    1  # nprocs
    inputs_base_3d_radiation_transport_packet_sampling  # inputs
    "analysis_packet_sampling.py --reference ../test_3d_radiation_transport_packet_sampling_seed42 --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (same-seed particle records are compared exactly)
    test_3d_radiation_transport_packet_sampling_seed42  # dependency
)

add_warpx_test(
    test_3d_radiation_transport_packet_sampling_seed43  # name
    3  # dims
    1  # nprocs
    "inputs_base_3d_radiation_transport_packet_sampling warpx.random_seed=43"  # inputs
    "analysis_packet_sampling.py --reference ../test_3d_radiation_transport_packet_sampling_seed42 --expect-different --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (different seeds must change the packet ensemble)
    test_3d_radiation_transport_packet_sampling_seed42  # dependency
)

add_warpx_test(
    test_3d_radiation_transport_multigroup_conversion  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_radiation_transport_multigroup_conversion  # inputs
    "analysis_rz.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)
