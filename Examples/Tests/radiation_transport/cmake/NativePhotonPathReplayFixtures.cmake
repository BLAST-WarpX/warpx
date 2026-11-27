# Minimal chronological native photon path gate.  The two-box 1D layout and
# three-rank MPI case exercise an empty rank without duplicating the owner's
# independent 3D corner fixture.
if(WarpX_LIB AND "1" IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL "DOUBLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_executable(test_native_photon_path_replay
        ${CMAKE_CURRENT_LIST_DIR}/../test_native_photon_path_replay.cpp)
    target_link_libraries(test_native_photon_path_replay PRIVATE lib_1d)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_native_photon_path_replay)
        target_compile_features(test_native_photon_path_replay PRIVATE cuda_std_20)
    else()
        target_compile_features(test_native_photon_path_replay PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_native_photon_path_replay)

    set(path_replay_dir
        "${CMAKE_CURRENT_BINARY_DIR}/native_photon_path_replay")
    file(MAKE_DIRECTORY "${path_replay_dir}")
    add_test(NAME test_1d_native_photon_path_replay
        COMMAND test_native_photon_path_replay
            ${CMAKE_CURRENT_LIST_DIR}/../inputs_base_1d_native_photon_collision_replay
            photons.num_particles_per_cell_each_dim=1)
    set_tests_properties(test_1d_native_photon_path_replay PROPERTIES
        WORKING_DIRECTORY "${path_replay_dir}" TIMEOUT 30
        ENVIRONMENT "OMP_NUM_THREADS=1")

    add_executable(test_native_photon_path_substeps
        ${CMAKE_CURRENT_LIST_DIR}/../test_native_photon_path_substeps.cpp)
    target_link_libraries(test_native_photon_path_substeps PRIVATE lib_1d)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_native_photon_path_substeps)
        target_compile_features(test_native_photon_path_substeps PRIVATE cuda_std_20)
    else()
        target_compile_features(test_native_photon_path_substeps PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_native_photon_path_substeps)
    set(path_substeps_dir
        "${CMAKE_CURRENT_BINARY_DIR}/native_photon_path_substeps")
    file(MAKE_DIRECTORY "${path_substeps_dir}")
    add_test(NAME test_1d_native_photon_path_substeps
        COMMAND test_native_photon_path_substeps
            ${CMAKE_CURRENT_LIST_DIR}/../inputs_base_1d_native_photon_collision_replay
            photons.num_particles_per_cell_each_dim=1)
    set_tests_properties(test_1d_native_photon_path_substeps PROPERTIES
        WORKING_DIRECTORY "${path_substeps_dir}" TIMEOUT 30
        ENVIRONMENT "OMP_NUM_THREADS=1")

    add_executable(test_native_photon_carry_routing
        ${CMAKE_CURRENT_LIST_DIR}/../test_native_photon_carry_routing.cpp)
    target_link_libraries(test_native_photon_carry_routing PRIVATE lib_1d)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_native_photon_carry_routing)
        target_compile_features(test_native_photon_carry_routing PRIVATE cuda_std_20)
    else()
        target_compile_features(test_native_photon_carry_routing PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_native_photon_carry_routing)
    set(carry_routing_dir
        "${CMAKE_CURRENT_BINARY_DIR}/native_photon_carry_routing")
    file(MAKE_DIRECTORY "${carry_routing_dir}")
    add_test(NAME test_1d_native_photon_carry_routing
        COMMAND test_native_photon_carry_routing
            ${CMAKE_CURRENT_LIST_DIR}/../inputs_base_1d_native_photon_collision_replay
            photons.num_particles_per_cell_each_dim=4)
    set_tests_properties(test_1d_native_photon_carry_routing PROPERTIES
        WORKING_DIRECTORY "${carry_routing_dir}" TIMEOUT 30
        ENVIRONMENT "OMP_NUM_THREADS=1")

    if(WarpX_MPI)
        foreach(ranks IN ITEMS 2 3)
            set(carry_routing_mpi_dir
                "${CMAKE_CURRENT_BINARY_DIR}/native_photon_carry_routing_mpi_${ranks}")
            file(MAKE_DIRECTORY "${carry_routing_mpi_dir}")
            add_test(NAME test_1d_native_photon_carry_routing_mpi_${ranks} COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_native_photon_carry_routing> ${MPIEXEC_POSTFLAGS}
                ${CMAKE_CURRENT_LIST_DIR}/../inputs_base_1d_native_photon_collision_replay
                photons.num_particles_per_cell_each_dim=4)
            set_tests_properties(test_1d_native_photon_carry_routing_mpi_${ranks}
                PROPERTIES PROCESSORS ${ranks} TIMEOUT 30
                WORKING_DIRECTORY "${carry_routing_mpi_dir}"
                ENVIRONMENT "OMP_NUM_THREADS=1")

            set(carry_distributed_dir
                "${CMAKE_CURRENT_BINARY_DIR}/native_photon_carry_distributed_mpi_${ranks}")
            file(MAKE_DIRECTORY "${carry_distributed_dir}")
            add_test(NAME test_1d_native_photon_carry_distributed_mpi_${ranks} COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_native_photon_carry_routing> ${MPIEXEC_POSTFLAGS}
                ${CMAKE_CURRENT_LIST_DIR}/../inputs_base_1d_native_photon_collision_replay
                photons.num_particles_per_cell_each_dim=4
                routing_test.distributed=1)
            set_tests_properties(test_1d_native_photon_carry_distributed_mpi_${ranks}
                PROPERTIES PROCESSORS ${ranks} TIMEOUT 30
                WORKING_DIRECTORY "${carry_distributed_dir}"
                ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()

        set(path_substeps_mpi_2_dir
            "${CMAKE_CURRENT_BINARY_DIR}/native_photon_path_substeps_mpi_2")
        file(MAKE_DIRECTORY "${path_substeps_mpi_2_dir}")
        add_test(NAME test_1d_native_photon_path_substeps_mpi_2 COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:test_native_photon_path_substeps> ${MPIEXEC_POSTFLAGS}
            ${CMAKE_CURRENT_LIST_DIR}/../inputs_base_1d_native_photon_collision_replay
            photons.num_particles_per_cell_each_dim=1)
        set_tests_properties(test_1d_native_photon_path_substeps_mpi_2 PROPERTIES
            PROCESSORS 2 TIMEOUT 30
            WORKING_DIRECTORY "${path_substeps_mpi_2_dir}"
            ENVIRONMENT "OMP_NUM_THREADS=1")

        set(path_replay_mpi_dir
            "${CMAKE_CURRENT_BINARY_DIR}/native_photon_path_replay_mpi_3")
        file(MAKE_DIRECTORY "${path_replay_mpi_dir}")
        add_test(NAME test_1d_native_photon_path_replay_mpi_3 COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:test_native_photon_path_replay> ${MPIEXEC_POSTFLAGS}
            ${CMAKE_CURRENT_LIST_DIR}/../inputs_base_1d_native_photon_collision_replay
            photons.num_particles_per_cell_each_dim=1)
        set_tests_properties(test_1d_native_photon_path_replay_mpi_3 PROPERTIES
            PROCESSORS 3 TIMEOUT 30 WORKING_DIRECTORY "${path_replay_mpi_dir}"
            ENVIRONMENT "OMP_NUM_THREADS=1")

        set(path_substeps_mpi_dir
            "${CMAKE_CURRENT_BINARY_DIR}/native_photon_path_substeps_mpi_3")
        file(MAKE_DIRECTORY "${path_substeps_mpi_dir}")
        add_test(NAME test_1d_native_photon_path_substeps_mpi_3 COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:test_native_photon_path_substeps> ${MPIEXEC_POSTFLAGS}
            ${CMAKE_CURRENT_LIST_DIR}/../inputs_base_1d_native_photon_collision_replay
            photons.num_particles_per_cell_each_dim=1)
        set_tests_properties(test_1d_native_photon_path_substeps_mpi_3 PROPERTIES
            PROCESSORS 3 TIMEOUT 30
            WORKING_DIRECTORY "${path_substeps_mpi_dir}"
            ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
endif()

# Cartesian XZ primitive control and separately gated substep activation.
if(WarpX_LIB AND "2" IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL "DOUBLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_executable(test_xz_native_photon_substeps
        ${CMAKE_CURRENT_LIST_DIR}/../test_xz_native_photon_substeps.cpp)
    target_link_libraries(test_xz_native_photon_substeps PRIVATE lib_2d)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_xz_native_photon_substeps)
        target_compile_features(test_xz_native_photon_substeps PRIVATE cuda_std_20)
    else()
        target_compile_features(test_xz_native_photon_substeps PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_xz_native_photon_substeps)
    foreach(mode IN ITEMS primitive helper)
        set(helper_flag 0)
        if(mode STREQUAL "helper")
            set(helper_flag 1)
        endif()
        set(xz_serial_dir
            "${CMAKE_CURRENT_BINARY_DIR}/xz_native_photon_${mode}")
        file(MAKE_DIRECTORY "${xz_serial_dir}")
        add_test(NAME test_2d_native_photon_substeps_${mode}
            COMMAND test_xz_native_photon_substeps
                ${CMAKE_CURRENT_LIST_DIR}/../inputs_base_xz_native_photon_substeps
                xz_test.require_helper=${helper_flag})
        set_tests_properties(test_2d_native_photon_substeps_${mode} PROPERTIES
            WORKING_DIRECTORY "${xz_serial_dir}" TIMEOUT 30
            ENVIRONMENT "OMP_NUM_THREADS=1")
        if(WarpX_MPI)
            foreach(ranks IN ITEMS 2 3)
                set(xz_mpi_dir
                    "${CMAKE_CURRENT_BINARY_DIR}/xz_native_photon_${mode}_mpi_${ranks}")
                file(MAKE_DIRECTORY "${xz_mpi_dir}")
                add_test(NAME test_2d_native_photon_substeps_${mode}_mpi_${ranks}
                    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                        ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:test_xz_native_photon_substeps>
                        ${MPIEXEC_POSTFLAGS}
                        ${CMAKE_CURRENT_LIST_DIR}/../inputs_base_xz_native_photon_substeps
                        xz_test.require_helper=${helper_flag})
                set_tests_properties(
                    test_2d_native_photon_substeps_${mode}_mpi_${ranks}
                    PROPERTIES PROCESSORS ${ranks} TIMEOUT 30
                    WORKING_DIRECTORY "${xz_mpi_dir}"
                    ENVIRONMENT "OMP_NUM_THREADS=1")
            endforeach()
        endif()
    endforeach()
    foreach(narrow_axis IN ITEMS x z)
        set(xz_aniso_input
            ${CMAKE_CURRENT_LIST_DIR}/../inputs_base_xz_native_photon_anisotropic_${narrow_axis})
        set(xz_aniso_dir
            "${CMAKE_CURRENT_BINARY_DIR}/xz_native_photon_anisotropic_${narrow_axis}")
        file(MAKE_DIRECTORY "${xz_aniso_dir}")
        add_test(NAME test_2d_native_photon_anisotropic_${narrow_axis}
            COMMAND test_xz_native_photon_substeps
                ${xz_aniso_input} xz_test.anisotropic=1)
        set_tests_properties(test_2d_native_photon_anisotropic_${narrow_axis}
            PROPERTIES WORKING_DIRECTORY "${xz_aniso_dir}" TIMEOUT 30
            ENVIRONMENT "OMP_NUM_THREADS=1")
        if(WarpX_MPI)
            foreach(ranks IN ITEMS 2 3)
                set(xz_aniso_mpi_dir
                    "${CMAKE_CURRENT_BINARY_DIR}/xz_native_photon_anisotropic_${narrow_axis}_mpi_${ranks}")
                file(MAKE_DIRECTORY "${xz_aniso_mpi_dir}")
                add_test(NAME test_2d_native_photon_anisotropic_${narrow_axis}_mpi_${ranks}
                    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                        ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:test_xz_native_photon_substeps>
                        ${MPIEXEC_POSTFLAGS}
                        ${xz_aniso_input} xz_test.anisotropic=1)
                set_tests_properties(
                    test_2d_native_photon_anisotropic_${narrow_axis}_mpi_${ranks}
                    PROPERTIES PROCESSORS ${ranks} TIMEOUT 30
                    WORKING_DIRECTORY "${xz_aniso_mpi_dir}"
                    ENVIRONMENT "OMP_NUM_THREADS=1")
            endforeach()
        endif()
    endforeach()
endif()
