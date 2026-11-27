# Native finite-mass packet/material fixed-point fixture. The callback is
# deliberately in-process so its photon snapshots and RNG directions stay frozen.
if(WarpX_LIB AND "1" IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL "DOUBLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_executable(test_packet_material_work_iteration
        test_packet_material_work_iteration.cpp)
    target_link_libraries(test_packet_material_work_iteration PRIVATE lib_1d)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_packet_material_work_iteration)
        target_compile_features(test_packet_material_work_iteration PRIVATE cuda_std_20)
    else()
        target_compile_features(test_packet_material_work_iteration PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_packet_material_work_iteration)

    set(packet_work_dir "${CMAKE_CURRENT_BINARY_DIR}/packet_material_work_iteration")
    file(MAKE_DIRECTORY "${packet_work_dir}")
    add_test(NAME test_1d_packet_material_work_iteration
        COMMAND test_packet_material_work_iteration
            inputs_base_1d_packet_material_work_iteration)
    set_tests_properties(test_1d_packet_material_work_iteration PROPERTIES
        WORKING_DIRECTORY "${packet_work_dir}"
        TIMEOUT 30
        ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
    foreach(case IN ITEMS strong omp)
        set(packet_work_threads 1)
        if(case STREQUAL "omp")
            set(packet_work_threads 2)
        endif()
        file(MAKE_DIRECTORY "${packet_work_dir}_${case}")
        add_test(NAME test_1d_packet_material_work_iteration_${case}
            COMMAND test_packet_material_work_iteration
                inputs_base_1d_packet_material_work_iteration test.recoil_fraction=0.2)
        set_tests_properties(test_1d_packet_material_work_iteration_${case} PROPERTIES
            WORKING_DIRECTORY "${packet_work_dir}_${case}" TIMEOUT 30
            PROCESSORS ${packet_work_threads}
            ENVIRONMENT "OMP_NUM_THREADS=${packet_work_threads};AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
    endforeach()
    if(WarpX_MPI)
        set(packet_work_mpi_dir "${CMAKE_CURRENT_BINARY_DIR}/packet_material_work_iteration_mpi")
        file(MAKE_DIRECTORY "${packet_work_mpi_dir}")
        add_test(NAME test_1d_packet_material_work_iteration_mpi COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:test_packet_material_work_iteration> ${MPIEXEC_POSTFLAGS}
            inputs_base_1d_packet_material_work_iteration)
        set_tests_properties(test_1d_packet_material_work_iteration_mpi PROPERTIES
            PROCESSORS 2 WORKING_DIRECTORY "${packet_work_mpi_dir}" TIMEOUT 30
            ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
        foreach(ranks IN ITEMS 2 3)
            file(MAKE_DIRECTORY "${packet_work_mpi_dir}_strong_${ranks}")
            add_test(NAME test_1d_packet_material_work_iteration_strong_mpi_${ranks} COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_packet_material_work_iteration> ${MPIEXEC_POSTFLAGS}
                inputs_base_1d_packet_material_work_iteration test.recoil_fraction=0.2)
            set_tests_properties(test_1d_packet_material_work_iteration_strong_mpi_${ranks}
                PROPERTIES PROCESSORS ${ranks} TIMEOUT 30
                WORKING_DIRECTORY "${packet_work_mpi_dir}_strong_${ranks}"
                ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
        endforeach()
    endif()
    foreach(ranks IN ITEMS 1 3)
        if(ranks GREATER 1 AND NOT WarpX_MPI)
            continue()
        endif()
        set(command $<TARGET_FILE:test_packet_material_work_iteration>)
        if(ranks GREATER 1)
            set(command ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_packet_material_work_iteration>
                ${MPIEXEC_POSTFLAGS})
        endif()
        set(test test_1d_packet_material_work_mixture_mpi${ranks})
        file(MAKE_DIRECTORY ${packet_work_dir}/${test})
        add_test(NAME ${test} COMMAND ${command}
            inputs_base_1d_packet_material_work_iteration_mixture)
        set_tests_properties(${test} PROPERTIES TIMEOUT 30 PROCESSORS ${ranks}
            WORKING_DIRECTORY ${packet_work_dir}/${test}
            ENVIRONMENT "OMP_NUM_THREADS=1;AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")
    endforeach()
endif()
