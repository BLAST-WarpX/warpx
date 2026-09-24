if(WarpX_LIB AND "1" IN_LIST WarpX_DIMS)
    add_executable(test_particle_energy_precision test_particle_energy_precision.cpp)
    target_link_libraries(test_particle_energy_precision PRIVATE lib_1d)
    target_compile_features(test_particle_energy_precision PRIVATE cxx_std_20)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_particle_energy_precision)
        target_compile_features(test_particle_energy_precision PRIVATE cuda_std_20)
    endif()
    warpx_set_compile_warnings(test_particle_energy_precision)
    add_test(NAME test_1d_particle_energy_precision COMMAND test_particle_energy_precision
        ${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_particle_energy_precision
        amrex.the_arena_init_size=0)
    set_tests_properties(test_1d_particle_energy_precision PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 60)
endif()
