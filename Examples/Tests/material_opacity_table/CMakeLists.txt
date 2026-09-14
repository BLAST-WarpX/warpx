# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

if(WarpX_LIB)
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
