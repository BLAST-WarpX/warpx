if(WarpX_LIB AND "1" IN_LIST WarpX_DIMS)
    add_executable(test_plotfile_particle_state test_plotfile_particle_state.cpp)
    target_link_libraries(test_plotfile_particle_state PRIVATE lib_1d)
    target_compile_features(test_plotfile_particle_state PRIVATE cxx_std_20)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_plotfile_particle_state)
        target_compile_features(test_plotfile_particle_state PRIVATE cuda_std_20)
    endif()
    warpx_set_compile_warnings(test_plotfile_particle_state)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output)
    add_test(NAME test_1d_plotfile_particle_state
        COMMAND test_plotfile_particle_state
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_plotfile_particle_state
            amrex.the_arena_init_size=0)
    set_tests_properties(test_1d_plotfile_particle_state PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output
        ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 60
        FIXTURES_SETUP plotfile_particle_state)
    add_test(NAME test_1d_plotfile_particle_state_output
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis.py)
    set_tests_properties(test_1d_plotfile_particle_state_output PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output
        FIXTURES_REQUIRED plotfile_particle_state TIMEOUT 30)
endif()
