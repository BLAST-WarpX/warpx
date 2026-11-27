# Integration-owned snippet: include after PacketDiffusionNativeInterval.cpp and
# its field/packet prerequisites are registered in lib_1d. Not included in the
# primitive-only task branch, where those production files have not landed.
if(WarpX_LIB AND "1" IN_LIST WarpX_DIMS AND WarpX_PRECISION STREQUAL "DOUBLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_executable(test_packet_diffusion_native_interval test_packet_diffusion_native_interval.cpp)
    target_link_libraries(test_packet_diffusion_native_interval PRIVATE lib_1d)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_packet_diffusion_native_interval)
        target_compile_features(test_packet_diffusion_native_interval PRIVATE cuda_std_20)
    else()
        target_compile_features(test_packet_diffusion_native_interval PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_packet_diffusion_native_interval)
    foreach(threads IN ITEMS 1 2)
        set(interval_test test_1d_packet_diffusion_native_interval_omp_${threads})
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${interval_test}")
        add_test(NAME ${interval_test} COMMAND test_packet_diffusion_native_interval
            "${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_1d_packet_diffusion_native_interval")
        set_tests_properties(${interval_test} PROPERTIES PROCESSORS ${threads} TIMEOUT 45
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${interval_test}"
            ENVIRONMENT "OMP_NUM_THREADS=${threads}")
    endforeach()
    set(interval_collision_test test_1d_packet_diffusion_native_interval_collisions)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${interval_collision_test}")
    add_test(NAME ${interval_collision_test} COMMAND test_packet_diffusion_native_interval
        "${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_1d_packet_diffusion_native_interval"
        test.kinetic_collisions=1)
    set_tests_properties(${interval_collision_test} PROPERTIES TIMEOUT 45
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${interval_collision_test}"
        ENVIRONMENT "OMP_NUM_THREADS=1")
    foreach(case IN ITEMS normal oblique asymptotic collisions retained)
        set(traction_case test_1d_packet_diffusion_quasi_static_${case})
        set(traction_options test.quasi_static_angular_traction=1)
        if(NOT case STREQUAL "normal")
            list(APPEND traction_options
                "photons.multiple_particles_ux=0.3 0.3"
                "photons.multiple_particles_uy=0.4 0.4"
                "photons.multiple_particles_uz=0.8660254037844386 -0.8660254037844386")
        endif()
        if(case STREQUAL "asymptotic")
            list(APPEND traction_options test.interface_lambda=0.7104)
        elseif(case STREQUAL "collisions")
            list(APPEND traction_options test.kinetic_collisions=1 test.proper_rates=1)
        elseif(case STREQUAL "retained")
            list(APPEND traction_options test.retain_constitutive_momentum=1)
        endif()
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${traction_case}")
        add_test(NAME ${traction_case} COMMAND test_packet_diffusion_native_interval
            "${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_1d_packet_diffusion_native_interval"
            ${traction_options})
        set_tests_properties(${traction_case} PROPERTIES TIMEOUT 45
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${traction_case}"
            ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
    foreach(case IN ITEMS proper_rates asymptotic_boundary)
        set(interval_case test_1d_packet_diffusion_native_interval_${case})
        if(case STREQUAL "proper_rates")
            set(case_options test.kinetic_collisions=1 test.proper_rates=1)
        else()
            set(case_options test.interface_lambda=0.7104)
        endif()
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${interval_case}")
        add_test(NAME ${interval_case} COMMAND test_packet_diffusion_native_interval
            "${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_1d_packet_diffusion_native_interval"
            ${case_options})
        set_tests_properties(${interval_case} PROPERTIES TIMEOUT 45
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${interval_case}"
            ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
    if(WarpX_MPI)
        foreach(ranks IN ITEMS 2 3)
            set(traction_case test_1d_packet_diffusion_quasi_static_mpi_${ranks})
            file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${traction_case}")
            add_test(NAME ${traction_case} COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_packet_diffusion_native_interval> ${MPIEXEC_POSTFLAGS}
                "${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_1d_packet_diffusion_native_interval"
                test.quasi_static_angular_traction=1
                "photons.multiple_particles_ux=0.3 0.3"
                "photons.multiple_particles_uy=0.4 0.4"
                "photons.multiple_particles_uz=0.8660254037844386 -0.8660254037844386")
            set_tests_properties(${traction_case} PROPERTIES PROCESSORS ${ranks} TIMEOUT 45
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${traction_case}"
                ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
        foreach(ranks IN ITEMS 2 3)
            set(interval_test test_1d_packet_diffusion_native_interval_mpi_${ranks})
            file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${interval_test}")
            add_test(NAME ${interval_test} COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_packet_diffusion_native_interval> ${MPIEXEC_POSTFLAGS}
                "${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_1d_packet_diffusion_native_interval")
            set_tests_properties(${interval_test} PROPERTIES PROCESSORS ${ranks} TIMEOUT 45
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${interval_test}"
                ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
        set(interval_collision_test test_1d_packet_diffusion_native_interval_collisions_mpi)
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${interval_collision_test}")
        add_test(NAME ${interval_collision_test} COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:test_packet_diffusion_native_interval> ${MPIEXEC_POSTFLAGS}
            "${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_1d_packet_diffusion_native_interval"
            test.kinetic_collisions=1)
        set_tests_properties(${interval_collision_test} PROPERTIES PROCESSORS 3 TIMEOUT 45
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${interval_collision_test}"
            ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
endif()
