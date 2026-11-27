if(WarpX_LIB AND RZ IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL DOUBLE)
    find_package(Python REQUIRED COMPONENTS Interpreter)
    add_executable(test_native_rz_magnetic_pressure test_native_rz_magnetic_pressure.cpp)
    target_link_libraries(test_native_rz_magnetic_pressure PRIVATE lib_rz)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_native_rz_magnetic_pressure)
        target_compile_features(test_native_rz_magnetic_pressure PRIVATE cuda_std_20)
    else()
        target_compile_features(test_native_rz_magnetic_pressure PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_native_rz_magnetic_pressure)
    add_executable(test_native_rz_pressure_runtime test_native_rz_pressure_runtime.cpp)
    target_link_libraries(test_native_rz_pressure_runtime PRIVATE lib_rz)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_native_rz_pressure_runtime)
        target_compile_features(test_native_rz_pressure_runtime PRIVATE cuda_std_20)
    else()
        target_compile_features(test_native_rz_pressure_runtime PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_native_rz_pressure_runtime)
    set(runtime_mpi_arguments)
    if(WarpX_MPI)
        set(runtime_mpi_arguments ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG})
    endif()
    foreach(grid IN ITEMS staggered collocated)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/runtime_${grid})
        add_test(NAME native_rz_pressure_runtime_${grid} COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_native_runtime.py
            $<TARGET_FILE:test_native_rz_pressure_runtime>
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_native_rz_pressure_runtime ${grid}
            ${runtime_mpi_arguments})
        set_tests_properties(native_rz_pressure_runtime_${grid} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/runtime_${grid}
            ENVIRONMENT "OMP_NUM_THREADS=1" PROCESSORS 2 TIMEOUT 120)
    endforeach()
    foreach(grid IN ITEMS staggered collocated)
        set(prefix native_rz_magnetic_pressure_)
        set(grid_arguments)
        set(folder_prefix)
        if(grid STREQUAL collocated)
            set(prefix native_rz_magnetic_pressure_collocated_)
            set(grid_arguments warpx.grid_type=collocated)
            set(folder_prefix collocated_)
        endif()
        foreach(case IN ITEMS serial single_box small_boxes mpi mpi_small_boxes)
            if(case MATCHES "^mpi" AND NOT WarpX_MPI)
                continue()
            endif()
            set(launcher)
            set(arguments)
            if(case MATCHES "^mpi")
                set(launcher ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS})
                if(case STREQUAL mpi_small_boxes)
                    set(arguments amr.max_grid_size=4)
                endif()
            elseif(case STREQUAL single_box)
                set(arguments amr.max_grid_size=16)
            elseif(case STREQUAL small_boxes)
                set(arguments amr.max_grid_size=4)
            endif()
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${folder_prefix}${case})
            add_test(NAME ${prefix}${case} COMMAND ${launcher}
                $<TARGET_FILE:test_native_rz_magnetic_pressure>
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_native_rz_magnetic_pressure
                ${arguments} ${grid_arguments})
            set_tests_properties(${prefix}${case} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${folder_prefix}${case}
                ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 60)
            if(case MATCHES "^mpi")
                set_tests_properties(${prefix}${case} PROPERTIES PROCESSORS 2)
            endif()
        endforeach()
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${folder_prefix}spatial)
        add_test(NAME ${prefix}spatial COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_spatial_convergence.py
            $<TARGET_FILE:test_native_rz_magnetic_pressure>
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_native_rz_magnetic_pressure ${grid_arguments})
        set_tests_properties(${prefix}spatial PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${folder_prefix}spatial
            ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 60)
    endforeach()
endif()
