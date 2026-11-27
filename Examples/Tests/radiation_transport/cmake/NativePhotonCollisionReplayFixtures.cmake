# Native coherent photon scattering replay fixture.  The 1D grid has two
# native boxes, so the three-rank MPI case intentionally contains an empty rank.
if(WarpX_LIB AND "1" IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL "DOUBLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_executable(test_native_photon_collision_replay
        test_native_photon_collision_replay.cpp)
    target_link_libraries(test_native_photon_collision_replay PRIVATE lib_1d)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_native_photon_collision_replay)
        target_compile_features(test_native_photon_collision_replay PRIVATE cuda_std_20)
    else()
        target_compile_features(test_native_photon_collision_replay PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_native_photon_collision_replay)

    add_executable(test_frozen_photon_collision_rate test_frozen_photon_collision_rate.cpp)
    target_link_libraries(test_frozen_photon_collision_rate PRIVATE lib_1d)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_frozen_photon_collision_rate)
        target_compile_features(test_frozen_photon_collision_rate PRIVATE cuda_std_20)
    else()
        target_compile_features(test_frozen_photon_collision_rate PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_frozen_photon_collision_rate)
    add_test(NAME test_1d_frozen_photon_collision_rate COMMAND test_frozen_photon_collision_rate)
    set_tests_properties(test_1d_frozen_photon_collision_rate PROPERTIES
        TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1")

    add_executable(test_native_photon_collision_bath test_native_photon_collision_bath.cpp)
    target_link_libraries(test_native_photon_collision_bath PRIVATE lib_1d)
    target_compile_features(test_native_photon_collision_bath PRIVATE cxx_std_20)
    warpx_set_compile_warnings(test_native_photon_collision_bath)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_native_photon_collision_bath)
        target_compile_features(test_native_photon_collision_bath PRIVATE cuda_std_20)
    endif()
    set(native_bath_dir "${CMAKE_CURRENT_BINARY_DIR}/native_photon_collision_bath")
    file(MAKE_DIRECTORY "${native_bath_dir}")
    add_test(NAME test_1d_native_photon_collision_bath COMMAND test_native_photon_collision_bath
        inputs_base_1d_native_photon_collision_replay photons.num_particles_per_cell_each_dim=16384)
    set_tests_properties(test_1d_native_photon_collision_bath PROPERTIES
        TIMEOUT 30 WORKING_DIRECTORY "${native_bath_dir}"
        ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")

    set(native_collision_dir
        "${CMAKE_CURRENT_BINARY_DIR}/native_photon_collision_replay")
    file(MAKE_DIRECTORY "${native_collision_dir}")
    add_test(NAME test_1d_native_photon_collision_replay
        COMMAND test_native_photon_collision_replay
            inputs_base_1d_native_photon_collision_replay)
    set_tests_properties(test_1d_native_photon_collision_replay PROPERTIES
        WORKING_DIRECTORY "${native_collision_dir}" TIMEOUT 30
        ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")

    set(native_collision_omp_dir
        "${CMAKE_CURRENT_BINARY_DIR}/native_photon_collision_replay_omp")
    file(MAKE_DIRECTORY "${native_collision_omp_dir}")
    add_test(NAME test_1d_native_photon_collision_replay_omp
        COMMAND test_native_photon_collision_replay
            inputs_base_1d_native_photon_collision_replay)
    set_tests_properties(test_1d_native_photon_collision_replay_omp PROPERTIES
        WORKING_DIRECTORY "${native_collision_omp_dir}" TIMEOUT 30 PROCESSORS 2
        ENVIRONMENT "OMP_NUM_THREADS=2;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")

    if(WarpX_MPI)
        set(native_bath_mpi_dir "${CMAKE_CURRENT_BINARY_DIR}/native_photon_collision_bath_mpi_3")
        file(MAKE_DIRECTORY "${native_bath_mpi_dir}")
        add_test(NAME test_1d_native_photon_collision_bath_mpi_3 COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:test_native_photon_collision_bath> ${MPIEXEC_POSTFLAGS}
            inputs_base_1d_native_photon_collision_replay
            photons.num_particles_per_cell_each_dim=16384)
        set_tests_properties(test_1d_native_photon_collision_bath_mpi_3 PROPERTIES
            PROCESSORS 3 TIMEOUT 30 WORKING_DIRECTORY "${native_bath_mpi_dir}"
            ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
        foreach(ranks IN ITEMS 1 2 3)
            set(native_collision_mpi_dir
                "${CMAKE_CURRENT_BINARY_DIR}/native_photon_collision_replay_mpi_${ranks}")
            file(MAKE_DIRECTORY "${native_collision_mpi_dir}")
            add_test(NAME test_1d_native_photon_collision_replay_mpi_${ranks} COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_native_photon_collision_replay>
                ${MPIEXEC_POSTFLAGS}
                inputs_base_1d_native_photon_collision_replay)
            set_tests_properties(test_1d_native_photon_collision_replay_mpi_${ranks}
                PROPERTIES PROCESSORS ${ranks} TIMEOUT 30
                WORKING_DIRECTORY "${native_collision_mpi_dir}"
                ENVIRONMENT
                    "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
        endforeach()
    endif()
endif()
