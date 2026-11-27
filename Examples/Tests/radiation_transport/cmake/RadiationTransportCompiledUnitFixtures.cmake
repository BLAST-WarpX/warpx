if(WarpX_LIB)
    if("1" IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL "DOUBLE"
       AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
        add_executable(test_packet_diffusion_leakage_faces test_packet_diffusion_leakage_faces.cpp)
        target_link_libraries(test_packet_diffusion_leakage_faces PRIVATE lib_1d)
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_packet_diffusion_leakage_faces)
            target_compile_features(test_packet_diffusion_leakage_faces PRIVATE cuda_std_20)
        else()
            target_compile_features(test_packet_diffusion_leakage_faces PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_packet_diffusion_leakage_faces)
        add_test(NAME test_packet_diffusion_leakage_faces COMMAND
            test_packet_diffusion_leakage_faces amrex.the_arena_init_size=0)
        set_tests_properties(test_packet_diffusion_leakage_faces PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 30)
        if(WarpX_MPI)
            foreach(leakage_ranks IN ITEMS 2 3)
                add_test(NAME test_packet_diffusion_leakage_faces_mpi_${leakage_ranks} COMMAND
                    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${leakage_ranks}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_packet_diffusion_leakage_faces>
                    ${MPIEXEC_POSTFLAGS} amrex.the_arena_init_size=0)
                set_tests_properties(test_packet_diffusion_leakage_faces_mpi_${leakage_ranks}
                    PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 30
                    PROCESSORS ${leakage_ranks})
            endforeach()
        endif()
    endif()
    if("1" IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL "DOUBLE"
       AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
        add_executable(test_photon_packet_transaction test_photon_packet_transaction.cpp)
        target_link_libraries(test_photon_packet_transaction PRIVATE lib_1d)
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_photon_packet_transaction)
            target_compile_features(test_photon_packet_transaction PRIVATE cuda_std_20)
        else()
            target_compile_features(test_photon_packet_transaction PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_photon_packet_transaction)
        add_executable(test_packet_emission_plan test_packet_emission_plan.cpp)
        target_link_libraries(test_packet_emission_plan PRIVATE lib_1d)
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_packet_emission_plan)
            target_compile_features(test_packet_emission_plan PRIVATE cuda_std_20)
        else()
            target_compile_features(test_packet_emission_plan PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_packet_emission_plan)
        add_executable(test_packet_diffusion_interface_native
            test_packet_diffusion_interface_native.cpp)
        target_link_libraries(test_packet_diffusion_interface_native PRIVATE lib_1d)
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_packet_diffusion_interface_native)
            target_compile_features(test_packet_diffusion_interface_native PRIVATE cuda_std_20)
        else()
            target_compile_features(test_packet_diffusion_interface_native PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_packet_diffusion_interface_native)
        foreach(emission_ranks IN ITEMS 1 2 3)
            if(emission_ranks GREATER 1 AND NOT WarpX_MPI)
                continue()
            endif()
            set(emission_name test_packet_emission_plan_${emission_ranks}rank)
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${emission_name})
            if(emission_ranks EQUAL 1)
                add_test(NAME ${emission_name} COMMAND test_packet_emission_plan
                    ${CMAKE_CURRENT_SOURCE_DIR}/inputs_1d_photon_packet_transaction)
            else()
                add_test(NAME ${emission_name} COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} ${emission_ranks} ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_packet_emission_plan> ${MPIEXEC_POSTFLAGS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/inputs_1d_photon_packet_transaction)
            endif()
            set_tests_properties(${emission_name} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${emission_name}
                PROCESSORS ${emission_ranks} ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 30)
        endforeach()
        foreach(interface_ranks IN ITEMS 1 2 3)
            if(interface_ranks GREATER 1 AND NOT WarpX_MPI)
                continue()
            endif()
            set(interface_name test_packet_diffusion_interface_native_${interface_ranks}rank)
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${interface_name})
            if(interface_ranks EQUAL 1)
                add_test(NAME ${interface_name} COMMAND test_packet_diffusion_interface_native
                    ${CMAKE_CURRENT_SOURCE_DIR}/inputs_1d_packet_diffusion_interface_native)
            else()
                add_test(NAME ${interface_name} COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} ${interface_ranks} ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_packet_diffusion_interface_native> ${MPIEXEC_POSTFLAGS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/inputs_1d_packet_diffusion_interface_native)
            endif()
            set_tests_properties(${interface_name} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${interface_name}
                PROCESSORS ${interface_ranks} ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 30)
        endforeach()
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/photon_transaction)
        add_test(NAME test_photon_packet_transaction COMMAND test_photon_packet_transaction
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_1d_photon_packet_transaction)
        set_tests_properties(test_photon_packet_transaction PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/photon_transaction
            PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=2" TIMEOUT 30)
        if(WarpX_MPI)
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/photon_transaction_mpi)
            add_test(NAME test_photon_packet_transaction_mpi COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_photon_packet_transaction> ${MPIEXEC_POSTFLAGS}
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_1d_photon_packet_transaction)
            set_tests_properties(test_photon_packet_transaction_mpi PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/photon_transaction_mpi
                PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 30)
        endif()
    endif()
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
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_coupled_spiner)
            target_compile_features(test_coupled_spiner PRIVATE cuda_std_20)
        endif()
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
            amrex.the_arena_init_size=0
            hybrid_pic_model.electron_eos_ions_table_file=${coupled_spiner_table})
        set_tests_properties(test_coupled_spiner PROPERTIES FIXTURES_REQUIRED coupled_spiner_table
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/coupled_spiner
            ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
        if(WarpX_MPI)
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/coupled_spiner_mpi)
            add_test(NAME test_coupled_spiner_mpi COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_coupled_spiner> inputs_base_1d_coupled_spiner
                amrex.the_arena_init_size=0
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
        add_executable(test_asymptotic_diffusion_interface
            asymptotic_diffusion_interface/test_asymptotic_diffusion_interface.cpp)
        add_executable(test_asymptotic_diffusion_work test_asymptotic_diffusion_work.cpp)
        foreach(diffusion_test IN ITEMS
                test_asymptotic_diffusion_interface test_asymptotic_diffusion_work)
            target_link_libraries(${diffusion_test} PRIVATE lib_${frame_suffix})
            target_compile_features(${diffusion_test} PRIVATE cxx_std_20)
            warpx_set_compile_warnings(${diffusion_test})
            add_test(NAME ${diffusion_test} COMMAND ${diffusion_test})
            set_tests_properties(${diffusion_test} PROPERTIES TIMEOUT 30)
        endforeach()
        add_executable(test_moment_transport_ledger test_moment_transport_ledger.cpp)
        target_link_libraries(test_moment_transport_ledger PRIVATE lib_${frame_suffix})
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_moment_transport_ledger)
            target_compile_features(test_moment_transport_ledger PRIVATE cuda_std_20)
        else()
            target_compile_features(test_moment_transport_ledger PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_moment_transport_ledger)
        add_test(NAME test_moment_transport_ledger COMMAND test_moment_transport_ledger
            amrex.the_arena_init_size=0)
        set_tests_properties(test_moment_transport_ledger PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
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
        add_executable(test_packet_moment_reconstruction test_packet_moment_reconstruction.cpp)
        target_link_libraries(test_packet_moment_reconstruction PRIVATE lib_${frame_suffix})
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_packet_moment_reconstruction)
            target_compile_features(test_packet_moment_reconstruction PRIVATE cuda_std_20)
        else()
            target_compile_features(test_packet_moment_reconstruction PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_packet_moment_reconstruction)
        add_test(NAME test_packet_moment_reconstruction COMMAND test_packet_moment_reconstruction
            amrex.throw_exception=1 amrex.signal_handling=0)
        set_tests_properties(test_packet_moment_reconstruction PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
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
        foreach(limited_dimension IN ITEMS 1 2 3)
            if(NOT limited_dimension IN_LIST WarpX_DIMS)
                continue()
            endif()
            warpx_set_suffix_dims(limited_suffix ${limited_dimension})
            set(limited_target test_${limited_suffix}_limited_moment_transport)
            add_executable(${limited_target} test_limited_moment_transport.cpp)
            target_link_libraries(${limited_target} PRIVATE lib_${limited_suffix})
            if(WarpX_COMPUTE STREQUAL "CUDA")
                setup_target_for_cuda_compilation(${limited_target})
                target_compile_features(${limited_target} PRIVATE cuda_std_20)
            else()
                target_compile_features(${limited_target} PRIVATE cxx_std_20)
            endif()
            warpx_set_compile_warnings(${limited_target})
            add_test(NAME ${limited_target} COMMAND ${limited_target}
                amrex.the_arena_init_size=0)
            set_tests_properties(${limited_target} PROPERTIES
                ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 30)
            if(WarpX_MPI)
                foreach(limited_ranks IN ITEMS 2 3)
                    add_test(NAME ${limited_target}_mpi_${limited_ranks} COMMAND
                        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${limited_ranks}
                        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${limited_target}>
                        ${MPIEXEC_POSTFLAGS} amrex.the_arena_init_size=0)
                    set_tests_properties(${limited_target}_mpi_${limited_ranks} PROPERTIES
                        ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 30
                        PROCESSORS ${limited_ranks})
                endforeach()
            endif()
        endforeach()
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
        if(WarpX_PRECISION STREQUAL "DOUBLE")
            add_executable(test_radiation_implicit_diffusion_reuse
                test_implicit_diffusion_reuse.cpp)
            target_link_libraries(test_radiation_implicit_diffusion_reuse PRIVATE lib_1d)
            target_compile_features(test_radiation_implicit_diffusion_reuse PRIVATE cxx_std_20)
            warpx_set_compile_warnings(test_radiation_implicit_diffusion_reuse)
            add_test(NAME test_radiation_implicit_diffusion_reuse
                COMMAND test_radiation_implicit_diffusion_reuse)
            set_tests_properties(test_radiation_implicit_diffusion_reuse PROPERTIES
                ENVIRONMENT "OMP_NUM_THREADS=1")
            add_executable(test_radiation_flux_relaxed_diffusion
                test_flux_relaxed_diffusion.cpp)
            target_link_libraries(test_radiation_flux_relaxed_diffusion PRIVATE lib_1d)
            target_compile_features(test_radiation_flux_relaxed_diffusion PRIVATE cxx_std_20)
            warpx_set_compile_warnings(test_radiation_flux_relaxed_diffusion)
            add_test(NAME test_radiation_flux_relaxed_diffusion
                COMMAND test_radiation_flux_relaxed_diffusion)
            set_tests_properties(test_radiation_flux_relaxed_diffusion PROPERTIES
                ENVIRONMENT "OMP_NUM_THREADS=1")
            if(WarpX_MPI)
                add_test(NAME test_radiation_implicit_diffusion_reuse_mpi COMMAND
                    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_radiation_implicit_diffusion_reuse> ${MPIEXEC_POSTFLAGS})
                set_tests_properties(test_radiation_implicit_diffusion_reuse_mpi PROPERTIES
                    PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1")
                add_test(NAME test_radiation_flux_relaxed_diffusion_mpi COMMAND
                    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_radiation_flux_relaxed_diffusion> ${MPIEXEC_POSTFLAGS})
                set_tests_properties(test_radiation_flux_relaxed_diffusion_mpi PROPERTIES
                    PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1")
            endif()
        endif()
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

    if(RZ IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL "DOUBLE"
       AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
        add_executable(test_rz_coupled_moment_source test_rz_coupled_moment_source.cpp)
        target_link_libraries(test_rz_coupled_moment_source PRIVATE lib_rz)
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_rz_coupled_moment_source)
            target_compile_features(test_rz_coupled_moment_source PRIVATE cuda_std_20)
        else()
            target_compile_features(test_rz_coupled_moment_source PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_rz_coupled_moment_source)
        foreach(coupled_case IN ITEMS drag thermal spatial rollback angular angular_lte angular_rollback)
            set(rz_coupled_args)
            if(coupled_case STREQUAL thermal)
                list(APPEND rz_coupled_args test.thermal=1)
            elseif(coupled_case STREQUAL spatial OR coupled_case STREQUAL rollback)
                list(APPEND rz_coupled_args test.spatial=1)
            elseif(coupled_case STREQUAL angular OR coupled_case STREQUAL angular_lte
                   OR coupled_case STREQUAL angular_rollback)
                list(APPEND rz_coupled_args test.angular=1 test.spatial=1)
                if(coupled_case STREQUAL angular_lte)
                    # Strong LTE feedback retains the damped source solve.
                    list(APPEND rz_coupled_args test.thermal=1)
                else()
                    list(APPEND rz_coupled_args test.relaxation=1)
                endif()
            endif()
            if(coupled_case STREQUAL spatial)
                # Qualified full-relaxation solve: identical physical gates,
                # fewer outer fixed-point iterations on CPU and CUDA.
                list(APPEND rz_coupled_args test.relaxation=1)
            endif()
            if(coupled_case STREQUAL rollback OR coupled_case STREQUAL angular_rollback)
                list(APPEND rz_coupled_args test.rollback=1 amr.max_grid_size=4)
            endif()
            set(rz_coupled_dir "${CMAKE_CURRENT_BINARY_DIR}/rz_coupled_${coupled_case}")
            file(MAKE_DIRECTORY ${rz_coupled_dir})
            add_test(NAME test_rz_coupled_moment_source_${coupled_case}
                COMMAND test_rz_coupled_moment_source inputs_rz_coupled_moment_source ${rz_coupled_args})
            set_tests_properties(test_rz_coupled_moment_source_${coupled_case} PROPERTIES
                WORKING_DIRECTORY ${rz_coupled_dir}
                ENVIRONMENT "OMP_NUM_THREADS=2;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
            if(WarpX_COMPUTE STREQUAL CUDA AND (coupled_case STREQUAL spatial OR coupled_case STREQUAL rollback
               OR coupled_case STREQUAL angular OR coupled_case STREQUAL angular_lte
               OR coupled_case STREQUAL angular_rollback))
                set_tests_properties(test_rz_coupled_moment_source_${coupled_case} PROPERTIES
                    TIMEOUT 5400 LABELS slow)
            endif()
        endforeach()
        if(WarpX_MPI)
            set(rz_angular_rollback_dir "${CMAKE_CURRENT_BINARY_DIR}/rz_coupled_angular_rollback_mpi")
            file(MAKE_DIRECTORY ${rz_angular_rollback_dir})
            add_test(NAME test_rz_coupled_moment_source_angular_rollback_mpi
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_rz_coupled_moment_source> ${MPIEXEC_POSTFLAGS}
                    inputs_rz_coupled_moment_source test.angular=1 test.spatial=1
                    test.rollback=1 test.relaxation=1 amr.max_grid_size=4)
            set_tests_properties(test_rz_coupled_moment_source_angular_rollback_mpi PROPERTIES
                PROCESSORS 2 RUN_SERIAL TRUE WORKING_DIRECTORY ${rz_angular_rollback_dir}
                ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
            if(WarpX_COMPUTE STREQUAL CUDA)
                set_tests_properties(test_rz_coupled_moment_source_angular_rollback_mpi PROPERTIES
                    TIMEOUT 5400 LABELS slow)
            endif()
            set(rz_angular_source_dir "${CMAKE_CURRENT_BINARY_DIR}/rz_coupled_angular_mpi")
            file(MAKE_DIRECTORY ${rz_angular_source_dir})
            add_test(NAME test_rz_coupled_moment_source_angular_mpi
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_rz_coupled_moment_source> ${MPIEXEC_POSTFLAGS}
                    inputs_rz_coupled_moment_source test.angular=1 test.spatial=1
                    test.relaxation=1 amr.max_grid_size=4)
            set_tests_properties(test_rz_coupled_moment_source_angular_mpi PROPERTIES
                PROCESSORS 2 RUN_SERIAL TRUE WORKING_DIRECTORY ${rz_angular_source_dir}
                ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
            if(WarpX_COMPUTE STREQUAL CUDA)
                set_tests_properties(test_rz_coupled_moment_source_angular_mpi PROPERTIES TIMEOUT 5400 LABELS slow)
            endif()
            set(rz_coupled_mpi_dir "${CMAKE_CURRENT_BINARY_DIR}/rz_coupled_rollback_mpi")
            file(MAKE_DIRECTORY ${rz_coupled_mpi_dir})
            add_test(NAME test_rz_coupled_moment_source_rollback_mpi
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_rz_coupled_moment_source> ${MPIEXEC_POSTFLAGS}
                    inputs_rz_coupled_moment_source test.spatial=1 test.rollback=1 amr.max_grid_size=4)
            set_tests_properties(test_rz_coupled_moment_source_rollback_mpi PROPERTIES
                PROCESSORS 2 RUN_SERIAL TRUE WORKING_DIRECTORY ${rz_coupled_mpi_dir}
                ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
            if(WarpX_COMPUTE STREQUAL CUDA)
                set_tests_properties(test_rz_coupled_moment_source_rollback_mpi PROPERTIES
                    TIMEOUT 5400 LABELS slow)
            endif()
        endif()
        add_executable(test_rz_angular_wall test_rz_angular_wall.cpp)
        target_link_libraries(test_rz_angular_wall PRIVATE lib_rz)
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_rz_angular_wall)
            target_compile_features(test_rz_angular_wall PRIVATE cuda_std_20)
        else()
            target_compile_features(test_rz_angular_wall PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_rz_angular_wall)
        set(rz_angular_wall_dir "${CMAKE_CURRENT_BINARY_DIR}/rz_angular_wall")
        file(MAKE_DIRECTORY ${rz_angular_wall_dir})
        add_test(NAME test_rz_angular_wall COMMAND test_rz_angular_wall inputs_base_rz_particle_impulse)
        set_tests_properties(test_rz_angular_wall PROPERTIES WORKING_DIRECTORY ${rz_angular_wall_dir}
            ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
        if(WarpX_MPI)
            add_test(NAME test_rz_angular_wall_mpi
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_rz_angular_wall> ${MPIEXEC_POSTFLAGS} inputs_base_rz_particle_impulse)
            set_tests_properties(test_rz_angular_wall_mpi PROPERTIES PROCESSORS 2 RUN_SERIAL TRUE
                WORKING_DIRECTORY ${rz_angular_wall_dir}
                ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
        endif()
        add_executable(test_rz_specular_trajectory test_rz_specular_trajectory.cpp)
        target_link_libraries(test_rz_specular_trajectory PRIVATE lib_rz)
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_rz_specular_trajectory)
            target_compile_features(test_rz_specular_trajectory PRIVATE cuda_std_20)
        else()
            target_compile_features(test_rz_specular_trajectory PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_rz_specular_trajectory)
        add_test(NAME test_rz_specular_trajectory COMMAND test_rz_specular_trajectory
            amrex.the_arena_init_size=0)
        set_tests_properties(test_rz_specular_trajectory PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
        add_executable(test_rz_angular_geometry test_rz_angular_geometry.cpp)
        target_link_libraries(test_rz_angular_geometry PRIVATE lib_rz)
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_rz_angular_geometry)
            target_compile_features(test_rz_angular_geometry PRIVATE cuda_std_20)
        else()
            target_compile_features(test_rz_angular_geometry PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_rz_angular_geometry)
        add_test(NAME test_rz_angular_geometry COMMAND test_rz_angular_geometry
            amrex.the_arena_init_size=0)
        set_tests_properties(test_rz_angular_geometry PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
        add_executable(test_rz_moment_transport test_rz_moment_transport.cpp)
        target_link_libraries(test_rz_moment_transport PRIVATE lib_rz)
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_rz_moment_transport)
            target_compile_features(test_rz_moment_transport PRIVATE cuda_std_20)
        else()
            target_compile_features(test_rz_moment_transport PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_rz_moment_transport)
        foreach(moment_case IN ITEMS uniform radial axial lte rotation rotating_material)
            add_test(NAME test_rz_moment_transport_${moment_case}
                COMMAND test_rz_moment_transport test.case=${moment_case}
                    amrex.the_arena_init_size=0)
            set_tests_properties(test_rz_moment_transport_${moment_case} PROPERTIES
                ENVIRONMENT "OMP_NUM_THREADS=2")
        endforeach()
        set(rz_moment_compression_dir "${CMAKE_CURRENT_BINARY_DIR}/rz_moment_compression")
        file(MAKE_DIRECTORY ${rz_moment_compression_dir})
        add_test(NAME test_rz_moment_transport_compression
            COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_rz_moment_compression.py
                $<TARGET_FILE:test_rz_moment_transport>)
        set_tests_properties(test_rz_moment_transport_compression PROPERTIES
            WORKING_DIRECTORY ${rz_moment_compression_dir} ENVIRONMENT "OMP_NUM_THREADS=2")
        if(WarpX_MPI)
            add_test(NAME test_rz_moment_transport_radial_mpi
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_rz_moment_transport> ${MPIEXEC_POSTFLAGS}
                    test.case=radial amrex.the_arena_init_size=0)
            set_tests_properties(test_rz_moment_transport_radial_mpi PROPERTIES
                PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1" RUN_SERIAL TRUE)
            add_test(NAME test_rz_moment_transport_angular_mpi
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_rz_moment_transport> ${MPIEXEC_POSTFLAGS}
                    test.case=rotating_material amrex.the_arena_init_size=0)
            set_tests_properties(test_rz_moment_transport_angular_mpi PROPERTIES
                PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1" RUN_SERIAL TRUE)
        endif()
        add_executable(test_rz_particle_impulse test_rz_particle_impulse.cpp)
        target_link_libraries(test_rz_particle_impulse PRIVATE lib_rz)
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_rz_particle_impulse)
            target_compile_features(test_rz_particle_impulse PRIVATE cuda_std_20)
        else()
            target_compile_features(test_rz_particle_impulse PRIVATE cxx_std_20)
        endif()
        warpx_set_compile_warnings(test_rz_particle_impulse)
        set(rz_angular_particle_dir "${CMAKE_CURRENT_BINARY_DIR}/rz_angular_particle")
        file(MAKE_DIRECTORY ${rz_angular_particle_dir})
        foreach(axis IN ITEMS 0 1)
            add_test(NAME test_rz_particle_nodal_impulse_axis_${axis}
                COMMAND test_rz_particle_impulse inputs_rz_nodal_particle_impulse
                    boundary.verboncoeur_axis_correction=${axis})
            set_tests_properties(test_rz_particle_nodal_impulse_axis_${axis} PROPERTIES
                WORKING_DIRECTORY ${rz_angular_particle_dir} TIMEOUT 60
                ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
            if(WarpX_MPI)
                add_test(NAME test_rz_particle_nodal_impulse_axis_${axis}_mpi
                    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:test_rz_particle_impulse> ${MPIEXEC_POSTFLAGS}
                        inputs_rz_nodal_particle_impulse boundary.verboncoeur_axis_correction=${axis})
                set_tests_properties(test_rz_particle_nodal_impulse_axis_${axis}_mpi PROPERTIES
                    PROCESSORS 3 RUN_SERIAL TRUE TIMEOUT 60 WORKING_DIRECTORY ${rz_angular_particle_dir}
                    ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
            endif()
        endforeach()
        foreach(order IN ITEMS 2 3)
            foreach(axis IN ITEMS 0 1)
                set(test test_rz_particle_nodal_shape${order}_axis${axis})
                add_test(NAME ${test} COMMAND test_rz_particle_impulse
                    inputs_rz_nodal_particle_impulse_high_order algo.particle_shape=${order}
                    boundary.verboncoeur_axis_correction=${axis})
                set_tests_properties(${test} PROPERTIES TIMEOUT 60
                    WORKING_DIRECTORY ${rz_angular_particle_dir}
                    ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                if(WarpX_MPI)
                    add_test(NAME ${test}_mpi COMMAND ${MPIEXEC_EXECUTABLE}
                        ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:test_rz_particle_impulse> ${MPIEXEC_POSTFLAGS}
                        inputs_rz_nodal_particle_impulse_high_order algo.particle_shape=${order}
                        boundary.verboncoeur_axis_correction=${axis})
                    set_tests_properties(${test}_mpi PROPERTIES TIMEOUT 60
                        PROCESSORS 3 RUN_SERIAL TRUE WORKING_DIRECTORY ${rz_angular_particle_dir}
                        ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                endif()
            endforeach()
        endforeach()
        foreach(order IN ITEMS 1 2 3)
            set(reflected_nodal_input inputs_rz_nodal_particle_impulse)
            if(order GREATER 1)
                set(reflected_nodal_input inputs_rz_nodal_particle_impulse_high_order)
            endif()
            foreach(axis IN ITEMS 0 1)
                set(test test_rz_particle_reflected_nodal_shape${order}_axis${axis})
                add_test(NAME ${test} COMMAND test_rz_particle_impulse
                    ${reflected_nodal_input} algo.particle_shape=${order}
                    test.reflecting_radial_density=1 boundary.verboncoeur_axis_correction=${axis})
                set_tests_properties(${test} PROPERTIES TIMEOUT 60
                    WORKING_DIRECTORY ${rz_angular_particle_dir}
                    ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                if(WarpX_MPI)
                    add_test(NAME ${test}_mpi COMMAND ${MPIEXEC_EXECUTABLE}
                        ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:test_rz_particle_impulse> ${MPIEXEC_POSTFLAGS}
                        ${reflected_nodal_input} algo.particle_shape=${order}
                        test.reflecting_radial_density=1 boundary.verboncoeur_axis_correction=${axis})
                    set_tests_properties(${test}_mpi PROPERTIES TIMEOUT 60
                        PROCESSORS 3 RUN_SERIAL TRUE WORKING_DIRECTORY ${rz_angular_particle_dir}
                        ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                endif()
            endforeach()
        endforeach()
        foreach(order IN ITEMS 1 2 3)
            set(closed_nodal_input inputs_rz_nodal_particle_impulse)
            if(order GREATER 1)
                set(closed_nodal_input inputs_rz_nodal_particle_impulse_high_order)
            endif()
            set(closed_nodal_options algo.particle_shape=${order}
                test.reflecting_radial_density=1 test.reflecting_axial_density=1
                "boundary.field_lo=none pec" "boundary.field_hi=pec pec"
                "boundary.particle_lo=none reflecting"
                "boundary.particle_hi=reflecting reflecting")
            foreach(axis IN ITEMS 0 1)
                set(test test_rz_particle_closed_nodal_shape${order}_axis${axis})
                add_test(NAME ${test} COMMAND test_rz_particle_impulse ${closed_nodal_input}
                    ${closed_nodal_options} boundary.verboncoeur_axis_correction=${axis})
                set_tests_properties(${test} PROPERTIES TIMEOUT 60
                    WORKING_DIRECTORY ${rz_angular_particle_dir}
                    ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                if(WarpX_MPI)
                    add_test(NAME ${test}_mpi COMMAND ${MPIEXEC_EXECUTABLE}
                        ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:test_rz_particle_impulse> ${MPIEXEC_POSTFLAGS}
                        ${closed_nodal_input} ${closed_nodal_options}
                        boundary.verboncoeur_axis_correction=${axis})
                    set_tests_properties(${test}_mpi PROPERTIES TIMEOUT 60
                        PROCESSORS 3 RUN_SERIAL TRUE WORKING_DIRECTORY ${rz_angular_particle_dir}
                        ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
                endif()
            endforeach()
        endforeach()
        add_test(NAME test_rz_particle_angular_impulse COMMAND test_rz_particle_impulse
            inputs_base_rz_particle_impulse test.angular_assignment=1 amrex.the_arena_init_size=0)
        set_tests_properties(test_rz_particle_angular_impulse PROPERTIES
            WORKING_DIRECTORY ${rz_angular_particle_dir}
            ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
        if(WarpX_MPI)
            add_test(NAME test_rz_particle_angular_impulse_mpi
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_rz_particle_impulse> ${MPIEXEC_POSTFLAGS}
                    inputs_base_rz_particle_impulse test.angular_assignment=1 amrex.the_arena_init_size=0)
            set_tests_properties(test_rz_particle_angular_impulse_mpi PROPERTIES
                PROCESSORS 2 RUN_SERIAL TRUE WORKING_DIRECTORY ${rz_angular_particle_dir}
                ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
        endif()
        set(rz_moving_environment
            "OMP_NUM_THREADS=2;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
        add_test(NAME test_radiation_checkpoint_process COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_checkpoint_process.py)
        function(add_rz_moving_runtime_case case_name input_file analysis_file)
            set(case_dir "${CMAKE_CURRENT_BINARY_DIR}/${case_name}")
            set(restart_dir "${CMAKE_CURRENT_BINARY_DIR}/${case_name}_restart")
            file(MAKE_DIRECTORY ${case_dir} ${restart_dir})
            add_test(NAME test_${case_name}.run COMMAND test_rz_particle_impulse ${input_file}
                test.radiation_runtime=1 test.moment_runtime=1 amrex.throw_exception=1
                amrex.abort_on_unused_inputs=1 amrex.the_arena_init_size=0)
            set_tests_properties(test_${case_name}.run PROPERTIES PROCESSORS 2
                WORKING_DIRECTORY ${case_dir} ENVIRONMENT "${rz_moving_environment}"
                FIXTURES_SETUP ${case_name}_checkpoint)
            add_test(NAME test_${case_name}.analysis COMMAND ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/${analysis_file} ${case_dir})
            set_tests_properties(test_${case_name}.analysis PROPERTIES
                FIXTURES_REQUIRED ${case_name}_checkpoint)
            set(restart_launcher)
            set(restart_postflags)
            if(WarpX_MPI)
                set(restart_launcher ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS})
                set(restart_postflags ${MPIEXEC_POSTFLAGS})
            endif()
            add_test(NAME test_${case_name}_restart.run COMMAND ${restart_launcher}
                $<TARGET_FILE:test_rz_particle_impulse> ${restart_postflags} ${input_file}
                test.radiation_runtime=1 test.moment_runtime=1 amr.restart=${case_dir}/diags/chk000128
                amrex.throw_exception=1 amrex.abort_on_unused_inputs=1 amrex.the_arena_init_size=0)
            set_tests_properties(test_${case_name}_restart.run PROPERTIES PROCESSORS 2
                WORKING_DIRECTORY ${restart_dir} ENVIRONMENT "${rz_moving_environment}"
                FIXTURES_REQUIRED ${case_name}_checkpoint FIXTURES_SETUP ${case_name}_restarted)
            if(WarpX_MPI)
                set_tests_properties(test_${case_name}_restart.run PROPERTIES
                    ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
            endif()
            add_test(NAME test_${case_name}_restart.analysis COMMAND ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/${analysis_file} ${restart_dir} --reference-directory ${case_dir})
            set_tests_properties(test_${case_name}_restart.analysis PROPERTIES
                FIXTURES_REQUIRED "${case_name}_checkpoint;${case_name}_restarted")
            if(WarpX_COMPUTE STREQUAL CUDA)
                set_tests_properties(test_${case_name}.run test_${case_name}_restart.run
                    PROPERTIES TIMEOUT 5400 LABELS slow)
            endif()
        endfunction()
        add_rz_moving_runtime_case(rz_moving_moment inputs_base_rz_moving_moment_pulse
            analysis_rz_moving_moment.py)
        add_rz_moving_runtime_case(rz_moving_compression inputs_base_rz_moving_moment_compression
            analysis_rz_moving_compression.py)
        add_rz_moving_runtime_case(rz_moving_rotation inputs_base_rz_moving_moment_rotation
            analysis_rz_moving_rotation.py)
        add_rz_moving_runtime_case(rz_moving_rotation_lte inputs_base_rz_moving_moment_rotation_lte
            analysis_rz_moving_rotation.py)
        foreach(corrupt IN ITEMS ledger meridional_model)
            set(rz_angular_corrupt_dir "${CMAKE_CURRENT_BINARY_DIR}/rz_moving_rotation_${corrupt}")
            file(MAKE_DIRECTORY ${rz_angular_corrupt_dir})
            add_test(NAME test_rz_moving_rotation_rejects_${corrupt} COMMAND ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/check_moving_moment_checkpoint.py
                $<TARGET_FILE:test_rz_particle_impulse>
                ${CMAKE_CURRENT_BINARY_DIR}/rz_moving_rotation/diags/chk000128
                ${corrupt} --angular)
            set_tests_properties(test_rz_moving_rotation_rejects_${corrupt} PROPERTIES
                WORKING_DIRECTORY ${rz_angular_corrupt_dir} ENVIRONMENT "${rz_moving_environment}"
                FIXTURES_REQUIRED rz_moving_rotation_checkpoint TIMEOUT 90)
        endforeach()
        set(rz_moving_dir "${CMAKE_CURRENT_BINARY_DIR}/rz_moving_moment")
        foreach(corrupt IN ITEMS ledger cartesian_model)
            set(rz_corrupt_dir "${CMAKE_CURRENT_BINARY_DIR}/rz_moving_moment_${corrupt}")
            file(MAKE_DIRECTORY ${rz_corrupt_dir})
            add_test(NAME test_rz_moving_moment_rejects_${corrupt} COMMAND ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/check_moving_moment_checkpoint.py
                $<TARGET_FILE:test_rz_particle_impulse> ${rz_moving_dir}/diags/chk000128
                ${corrupt} --rz)
            set_tests_properties(test_rz_moving_moment_rejects_${corrupt} PROPERTIES
                WORKING_DIRECTORY ${rz_corrupt_dir} ENVIRONMENT "${rz_moving_environment}"
                FIXTURES_REQUIRED rz_moving_moment_checkpoint TIMEOUT 90)
        endforeach()
        set(rz_impulse_dir "${CMAKE_CURRENT_BINARY_DIR}/rz_particle_impulse")
        file(MAKE_DIRECTORY ${rz_impulse_dir})
        foreach(input IN ITEMS inputs_base_rz_particle_impulse inputs_base_1d_particle_impulse)
            configure_file(${input} ${rz_impulse_dir}/${input} COPYONLY)
        endforeach()
        add_test(NAME test_rz_particle_impulse
            COMMAND test_rz_particle_impulse inputs_base_rz_particle_impulse)
        set_tests_properties(test_rz_particle_impulse PROPERTIES
            WORKING_DIRECTORY ${rz_impulse_dir}
            ENVIRONMENT "OMP_NUM_THREADS=2")
        add_test(NAME test_rz_particle_impulse_corners
            COMMAND test_rz_particle_impulse inputs_base_rz_particle_impulse
                "boundary.field_lo=none pec" "boundary.field_hi=pec pec"
                "boundary.particle_lo=none reflecting"
                "boundary.particle_hi=reflecting reflecting")
        set_tests_properties(test_rz_particle_impulse_corners PROPERTIES
            WORKING_DIRECTORY ${rz_impulse_dir}
            ENVIRONMENT "OMP_NUM_THREADS=2" RUN_SERIAL TRUE)
        if(WarpX_MPI)
            add_test(NAME test_rz_particle_impulse_mpi
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_rz_particle_impulse>
                    ${MPIEXEC_POSTFLAGS} inputs_base_rz_particle_impulse)
            set_tests_properties(test_rz_particle_impulse_mpi PROPERTIES
                WORKING_DIRECTORY ${rz_impulse_dir}
                PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1"
                RUN_SERIAL TRUE)
        endif()
        set(rz_carry_runtime_dir "${CMAKE_CURRENT_BINARY_DIR}/rz_carry_runtime")
        file(MAKE_DIRECTORY ${rz_carry_runtime_dir})
        foreach(input IN ITEMS inputs_base_rz_radiation_transport_particle_carry
                               inputs_test_rz_radiation_transport_absorption_momentum)
            configure_file(${input} ${rz_carry_runtime_dir}/${input} COPYONLY)
        endforeach()
        add_test(NAME test_rz_particle_carry_native_energy
            COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_rz_particle_carry.py
                $<TARGET_FILE:test_rz_particle_impulse> inputs_base_rz_radiation_transport_particle_carry
                test.radiation_runtime=1)
        set_tests_properties(test_rz_particle_carry_native_energy PROPERTIES
            WORKING_DIRECTORY ${rz_carry_runtime_dir} ENVIRONMENT "OMP_NUM_THREADS=2")
        foreach(unsupported IN ITEMS diffusion exact)
            if(unsupported STREQUAL diffusion)
                set(carry_reject_arg radiation_transport.enable_diffusion=1
                    radiation_transport.rosseland_transport_coefficient=10)
                set(carry_reject_message "RZ particle-owned radiation carry currently requires")
            else()
                set(carry_reject_arg radiation_transport.require_cell_interface_exact_streaming=1)
                set(carry_reject_message "Face-exact RZ streaming is incompatible")
            endif()
            add_test(NAME test_rz_particle_carry_rejects_${unsupported}
                COMMAND ${Python_EXECUTABLE}
                    ${WarpX_SOURCE_DIR}/Examples/Tests/implicit/check_command_output.py
                    --expect-failure --expect "${carry_reject_message}"
                    -- $<TARGET_FILE:test_rz_particle_impulse>
                    inputs_base_rz_radiation_transport_particle_carry
                    test.radiation_runtime=1 ${carry_reject_arg} amrex.throw_exception=1)
            set_tests_properties(test_rz_particle_carry_rejects_${unsupported} PROPERTIES
                WORKING_DIRECTORY ${rz_carry_runtime_dir}
                ENVIRONMENT "OMP_NUM_THREADS=1" RUN_SERIAL TRUE)
        endforeach()
        if(WarpX_MPI)
            add_test(NAME test_rz_particle_carry_native_energy_mpi
                COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_rz_particle_carry.py
                    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_rz_particle_impulse>
                    ${MPIEXEC_POSTFLAGS} inputs_base_rz_radiation_transport_particle_carry
                    test.radiation_runtime=1)
            set_tests_properties(test_rz_particle_carry_native_energy_mpi PROPERTIES
                WORKING_DIRECTORY ${rz_carry_runtime_dir}
                PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1" RUN_SERIAL TRUE)
        endif()
    endif()

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

    foreach(D IN ITEMS 1 2 3)
        if(D IN_LIST WarpX_DIMS)
            warpx_set_suffix_dims(SD ${D})
            set(cartesian_photon_target test_${SD}_cartesian_photon_boundary)
            add_executable(
                ${cartesian_photon_target}
                test_cartesian_photon_boundary.cpp)
            target_link_libraries(
                ${cartesian_photon_target}
                PRIVATE lib_${SD})
            if(WarpX_COMPUTE STREQUAL CUDA)
                setup_target_for_cuda_compilation(${cartesian_photon_target})
                target_compile_features(
                    ${cartesian_photon_target} PRIVATE cuda_std_20)
                set_target_properties(
                    ${cartesian_photon_target} PROPERTIES
                    CUDA_EXTENSIONS OFF
                    CUDA_STANDARD_REQUIRED ON)
            else()
                target_compile_features(
                    ${cartesian_photon_target} PRIVATE cxx_std_20)
                set_target_properties(
                    ${cartesian_photon_target} PROPERTIES
                    CXX_EXTENSIONS OFF
                    CXX_STANDARD_REQUIRED ON)
            endif()
            warpx_set_compile_warnings(${cartesian_photon_target})
            add_test(
                NAME ${cartesian_photon_target}
                COMMAND ${cartesian_photon_target})
        endif()
    endforeach()
endif()
