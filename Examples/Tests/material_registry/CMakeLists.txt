# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

if(WarpX_LIB)
    add_executable(test_material_registry test_material_registry.cpp)
    target_link_libraries(test_material_registry PRIVATE lib_${WarpX_DIMS_LAST})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_material_registry)
        target_compile_features(test_material_registry PRIVATE cuda_std_20)
        set_target_properties(
            test_material_registry PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON)
    else()
        target_compile_features(test_material_registry PRIVATE cxx_std_20)
        set_target_properties(
            test_material_registry PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON)
    endif()
    warpx_set_compile_warnings(test_material_registry)

    set(material_registry_common
        "materials.foam.species=foam"
        "materials.tungsten.species=tungsten"
        "materials.tungsten.electron_eos_table_file=tables/w.sp5"
        "materials.tungsten.electron_eos_material_id=74"
        "materials.tungsten.opacity_table_file=tables/w-opacity.h5"
        "materials.tungsten.opacity_material_id=74"
        "materials.tungsten.opacity_material_key=W"
        "test.expected_materials=foam tungsten"
        "amrex.abort_on_unused_inputs=1")

    add_test(
        NAME test_material_registry_order_a
        COMMAND test_material_registry
            "materials.names=tungsten foam"
            ${material_registry_common})
    add_test(
        NAME test_material_registry_order_b
        COMMAND test_material_registry
            "materials.names=foam tungsten"
            ${material_registry_common})

    add_test(
        NAME test_material_registry_duplicate_species_rejected
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:test_material_registry>
            -DTEST_CASE=duplicate_species
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)

    add_test(
        NAME test_material_registry_incomplete_handle_rejected
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:test_material_registry>
            -DTEST_CASE=incomplete_handle
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)

    add_test(
        NAME test_material_registry_missing_species_rejected
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:test_material_registry>
            -DTEST_CASE=missing_species
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)

    add_test(
        NAME test_material_registry_nonunique_dominance_rejected
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:test_material_registry>
            -DTEST_CASE=nonunique_dominance
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)

    add_test(
        NAME test_material_registry_capacity_rejected
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:test_material_registry>
            -DTEST_CASE=capacity
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)

    if(WarpX_APP AND TARGET app_1d)
        add_warpx_test(
            test_1d_material_registry_application_order_a  # name
            1  # dims
            1  # nprocs
            inputs_test_1d_material_registry_application_order_a  # inputs
            OFF  # analysis
            OFF  # checksum
            OFF  # dependency
        )
        add_warpx_test(
            test_1d_material_registry_application_order_b  # name
            1  # dims
            1  # nprocs
            inputs_test_1d_material_registry_application_order_b  # inputs
            OFF  # analysis
            OFF  # checksum
            OFF  # dependency
        )
        add_test(
            NAME test_material_registry_unknown_species_rejected
            COMMAND ${CMAKE_COMMAND}
                -DTEST_EXE=$<TARGET_FILE:app_1d>
                -DTEST_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/inputs_base_1d_material_registry
                -DTEST_CASE=unknown_species
                -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)
    endif()
endif()
