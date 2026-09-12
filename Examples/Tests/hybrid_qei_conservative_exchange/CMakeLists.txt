# Add tests (alphabetical order) ##############################################
#

list(GET WarpX_DIMS 0 source_noop_dim)
warpx_set_suffix_dims(source_noop_suffix ${source_noop_dim})
add_executable(test_electron_caloric_noop test_electron_caloric_noop.cpp)
target_link_libraries(test_electron_caloric_noop PRIVATE lib_${source_noop_suffix})
if(WarpX_COMPUTE STREQUAL CUDA)
    setup_target_for_cuda_compilation(test_electron_caloric_noop)
endif()
warpx_set_compile_warnings(test_electron_caloric_noop)
add_test(NAME test_electron_caloric_noop COMMAND test_electron_caloric_noop
    amrex.the_arena_init_size=0)
set_tests_properties(test_electron_caloric_noop PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")

add_warpx_test(
    test_2d_hybrid_qei_conservative_exchange  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_hybrid_qei_conservative_exchange  # inputs
    "analysis.py --precision ${WarpX_PRECISION} --particle-precision ${WarpX_PARTICLE_PRECISION}"  # analysis
    OFF  # checksum (analytic exchange and conservative ledgers are checked)
    OFF  # dependency
)
