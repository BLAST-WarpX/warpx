# Two native grids intentionally leave one of three MPI ranks empty.
if(WarpX_LIB AND "3" IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL "DOUBLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_executable(test_native_photon_path_corners test_native_photon_path_corners.cpp)
    target_link_libraries(test_native_photon_path_corners PRIVATE lib_3d)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_native_photon_path_corners)
        target_compile_features(test_native_photon_path_corners PRIVATE cuda_std_20)
    else()
        target_compile_features(test_native_photon_path_corners PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_native_photon_path_corners)
    set(path_corner_dir "${CMAKE_CURRENT_BINARY_DIR}/native_photon_path_corners")
    file(MAKE_DIRECTORY "${path_corner_dir}")
    add_test(NAME test_3d_native_photon_path_corners COMMAND test_native_photon_path_corners
        inputs_base_3d_native_photon_path_corners)
    set_tests_properties(test_3d_native_photon_path_corners PROPERTIES TIMEOUT 30
        WORKING_DIRECTORY "${path_corner_dir}"
        ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
    if(WarpX_MPI)
        set(path_corner_mpi_dir "${CMAKE_CURRENT_BINARY_DIR}/native_photon_path_corners_mpi_3")
        file(MAKE_DIRECTORY "${path_corner_mpi_dir}")
        add_test(NAME test_3d_native_photon_path_corners_mpi_3 COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:test_native_photon_path_corners> ${MPIEXEC_POSTFLAGS}
            inputs_base_3d_native_photon_path_corners)
        set_tests_properties(test_3d_native_photon_path_corners_mpi_3 PROPERTIES
            PROCESSORS 3 TIMEOUT 30 WORKING_DIRECTORY "${path_corner_mpi_dir}"
            ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
    endif()
endif()

# Cartesian 3D fixed packet-only subsegments. The base layout has two boxes;
# MPI3 intentionally leaves one rank without a box. The alternate fine layout
# has 16 boxes and tests the same native history on a second BoxArray.
if(WarpX_LIB AND "3" IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL "DOUBLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_executable(test_3d_native_photon_path_substeps
        ${CMAKE_CURRENT_LIST_DIR}/../test_3d_native_photon_path_substeps.cpp)
    target_link_libraries(test_3d_native_photon_path_substeps PRIVATE lib_3d)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_3d_native_photon_path_substeps)
        target_compile_features(test_3d_native_photon_path_substeps PRIVATE cuda_std_20)
    else()
        target_compile_features(test_3d_native_photon_path_substeps PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_3d_native_photon_path_substeps)
    set(path_substeps_two_input
        ${CMAKE_CURRENT_LIST_DIR}/../inputs_base_3d_native_photon_path_substeps)
    set(path_substeps_fine_input
        ${CMAKE_CURRENT_LIST_DIR}/../inputs_base_3d_native_photon_path_substeps_fine_boxes)
    if(WarpX_MPI)
        set(partition_dir "${CMAKE_CURRENT_BINARY_DIR}/substeps_3d_partition")
        file(MAKE_DIRECTORY "${partition_dir}")
        add_test(NAME test_3d_native_photon_substeps_changed_rank_restart COMMAND
            ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_LIST_DIR}/../analysis_3d_substep_partition.py
            $<TARGET_FILE:test_3d_native_photon_path_substeps>
            ${path_substeps_two_input} ${path_substeps_fine_input}
            --mpiexec=${MPIEXEC_EXECUTABLE} --numproc=${MPIEXEC_NUMPROC_FLAG})
        set_tests_properties(test_3d_native_photon_substeps_changed_rank_restart
            PROPERTIES WORKING_DIRECTORY "${partition_dir}" TIMEOUT 900
            PROCESSORS 3 ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
    foreach(ranks IN ITEMS 1 2 3)
        if(ranks GREATER 1 AND NOT WarpX_MPI)
            continue()
        endif()
        foreach(layout IN ITEMS two_boxes fine_boxes)
            if(layout STREQUAL "fine_boxes" AND ranks EQUAL 1)
                continue()
            endif()
            set(test test_3d_native_photon_path_substeps_${layout}_mpi${ranks})
            set(test_command $<TARGET_FILE:test_3d_native_photon_path_substeps>)
            if(layout STREQUAL "fine_boxes")
                set(test_input ${path_substeps_fine_input})
            else()
                set(test_input ${path_substeps_two_input})
            endif()
            if(ranks GREATER 1)
                set(test_command ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_3d_native_photon_path_substeps>
                    ${MPIEXEC_POSTFLAGS})
            endif()
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test})
            add_test(NAME ${test} COMMAND ${test_command} ${test_input})
            set_tests_properties(${test} PROPERTIES TIMEOUT 120 PROCESSORS ${ranks}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test}
                ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endif()
