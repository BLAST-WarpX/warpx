# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

if(WarpX_LIB)
    add_executable(test_text_opacity test_text_opacity.cpp)
    target_link_libraries(test_text_opacity PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_text_opacity)
        target_compile_features(test_text_opacity PRIVATE cuda_std_20)
    else()
        target_compile_features(test_text_opacity PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_text_opacity)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/text_opacity)
    add_test(NAME test_text_opacity COMMAND test_text_opacity amrex.throw_exception=1)
    set_tests_properties(test_text_opacity PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/text_opacity)

    if(NOT WarpX_MATERIAL_OPACITY_HDF5)
        return()
    endif()

    add_executable(test_material_opacity_response test_material_opacity_response.cpp)
    target_link_libraries(test_material_opacity_response PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_material_opacity_response)
        target_compile_features(test_material_opacity_response PRIVATE cuda_std_20)
    else()
        target_compile_features(test_material_opacity_response PRIVATE cxx_std_20)
    endif()
    warpx_set_compile_warnings(test_material_opacity_response)
    add_test(NAME material_opacity_response COMMAND test_material_opacity_response)
    set_tests_properties(material_opacity_response PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")

    add_executable(
        test_material_opacity_table
        test_material_opacity_table.cpp)
    target_link_libraries(
        test_material_opacity_table
        PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_material_opacity_table)
        target_compile_features(
            test_material_opacity_table PRIVATE cuda_std_20)
        set_target_properties(
            test_material_opacity_table PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON)
    else()
        target_compile_features(
            test_material_opacity_table PRIVATE cxx_std_20)
        set_target_properties(
            test_material_opacity_table PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON)
    endif()
    warpx_set_compile_warnings(test_material_opacity_table)

    add_test(
        NAME material_opacity_table
        COMMAND test_material_opacity_table)

    add_executable(
        test_material_opacity_zero
        test_material_opacity_zero.cpp)
    target_link_libraries(
        test_material_opacity_zero
        PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_material_opacity_zero)
        target_compile_features(
            test_material_opacity_zero PRIVATE cuda_std_20)
        set_target_properties(
            test_material_opacity_zero PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON)
    else()
        target_compile_features(
            test_material_opacity_zero PRIVATE cxx_std_20)
        set_target_properties(
            test_material_opacity_zero PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON)
    endif()
    warpx_set_compile_warnings(test_material_opacity_zero)

    foreach(case IN ITEMS contract reload consumers device)
        add_test(
            NAME material_opacity_zero_${case}
            COMMAND test_material_opacity_zero ${case})
        set_tests_properties(
            material_opacity_zero_${case}
            PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")
    endforeach()
    add_test(
        NAME material_opacity_zero_diffusion_guard
        COMMAND test_material_opacity_zero diffusion-guard)
    set_tests_properties(
        material_opacity_zero_diffusion_guard
        PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2" WILL_FAIL TRUE)

    set(material_opacity_runtime_fixture_dir
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/material_opacity_runtime_fixtures")
    add_test(
        NAME material_opacity_runtime_fixtures
        COMMAND test_material_opacity_table
            --write-runtime-fixtures
            "${material_opacity_runtime_fixture_dir}")
    set_tests_properties(
        material_opacity_runtime_fixtures
        PROPERTIES FIXTURES_SETUP material_opacity_runtime_fixtures)
endif()
