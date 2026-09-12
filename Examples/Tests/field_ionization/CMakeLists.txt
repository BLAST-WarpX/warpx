# Add tests (alphabetical order) ##############################################
#

if(WarpX_LIB)
    add_executable(
        test_field_ionization_transform
        test_field_ionization_transform.cpp)
    target_link_libraries(
        test_field_ionization_transform
        PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_field_ionization_transform)
        target_compile_features(
            test_field_ionization_transform PRIVATE cuda_std_20)
        set_target_properties(
            test_field_ionization_transform PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON)
    else()
        target_compile_features(
            test_field_ionization_transform PRIVATE cxx_std_20)
        set_target_properties(
            test_field_ionization_transform PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON)
    endif()
    warpx_set_compile_warnings(test_field_ionization_transform)

    add_test(
        NAME test_field_ionization_transform_component
        COMMAND test_field_ionization_transform)
endif()

add_warpx_test(
    test_2d_ionization_boost  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_ionization_boost  # inputs
    "analysis.py diags/diag1000420"  # analysis
    "analysis_default_regression.py --path diags/diag1000420"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ionization_lab  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_ionization_lab  # inputs
    "analysis.py diags/diag1001600"  # analysis
    "analysis_default_regression.py --path diags/diag1001600"  # checksum
    OFF  # dependency
)


add_warpx_test(
    test_2d_ionization_lab_restart  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_ionization_lab_restart  # inputs
    "analysis.py diags/diag1001600"  # analysis
    "analysis_default_regression.py --path diags/diag1001600 --rtol 1e-2"  # checksum
    test_2d_ionization_lab  # dependency
)

add_warpx_test(
    test_2d_ionization_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_ionization_picmi.py  # inputs
    "analysis.py diags/diag1001600"  # analysis
    "analysis_default_regression.py --path diags/diag1001600"  # checksum
    OFF  # dependency
)
