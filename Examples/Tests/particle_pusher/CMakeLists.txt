# Add tests (alphabetical order) ##############################################
#

if(WarpX_LIB)
    add_executable(
        test_boris_electric_work
        test_boris_electric_work.cpp)
    target_link_libraries(
        test_boris_electric_work
        PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_boris_electric_work)
        target_compile_features(test_boris_electric_work PRIVATE cuda_std_20)
        set_target_properties(
            test_boris_electric_work PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON)
    else()
        target_compile_features(test_boris_electric_work PRIVATE cxx_std_20)
        set_target_properties(
            test_boris_electric_work PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON)
    endif()
    warpx_set_compile_warnings(test_boris_electric_work)

    add_test(
        NAME test_boris_electric_work
        COMMAND test_boris_electric_work)

    if(WarpX_DIMS_LAST STREQUAL "1d" OR
       WarpX_DIMS_LAST STREQUAL "2d" OR
       WarpX_DIMS_LAST STREQUAL "3d")
        add_executable(
            test_boris_work_current_transpose
            test_boris_work_current_transpose.cpp)
        target_link_libraries(
            test_boris_work_current_transpose
            PRIVATE lib_${WarpX_DIMS_LAST})
        if(WarpX_COMPUTE STREQUAL CUDA)
            setup_target_for_cuda_compilation(test_boris_work_current_transpose)
            target_compile_features(test_boris_work_current_transpose PRIVATE cuda_std_20)
            set_target_properties(
                test_boris_work_current_transpose PROPERTIES
                CUDA_EXTENSIONS OFF
                CUDA_STANDARD_REQUIRED ON)
        else()
            target_compile_features(test_boris_work_current_transpose PRIVATE cxx_std_20)
            set_target_properties(
                test_boris_work_current_transpose PROPERTIES
                CXX_EXTENSIONS OFF
                CXX_STANDARD_REQUIRED ON)
        endif()
        warpx_set_compile_warnings(test_boris_work_current_transpose)

        add_test(
            NAME test_boris_work_current_transpose
            COMMAND test_boris_work_current_transpose)
    endif()
endif()

add_warpx_test(
    test_3d_particle_pusher  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_particle_pusher  # inputs
    "analysis.py diags/diag1010000"  # analysis
    "analysis_default_regression.py --path diags/diag1010000"  # checksum
    OFF  # dependency
)
