# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

if(TEST_CASE STREQUAL "duplicate_species")
    set(test_arguments
        "materials.names=foam tungsten"
        "materials.foam.species=ions"
        "materials.tungsten.species=ions"
        "test.expected_materials=foam tungsten")
    set(expected_text "duplicate mapping for species 'ions'")
elseif(TEST_CASE STREQUAL "incomplete_handle")
    set(test_arguments
        "materials.names=tungsten"
        "materials.tungsten.species=tungsten"
        "materials.tungsten.opacity_table_file=tables/w-opacity.h5"
        "test.expected_materials=tungsten")
    set(expected_text "opacity_material_id must be specified together")
elseif(TEST_CASE STREQUAL "missing_species")
    set(test_arguments
        "materials.names=tungsten"
        "test.expected_materials=tungsten")
    set(expected_text "materials.tungsten.species not found in database")
elseif(TEST_CASE STREQUAL "nonunique_dominance")
    set(test_arguments
        "materials.names=tungsten"
        "materials.tungsten.species=tungsten"
        "materials.mixed_cell_relative_tolerance=0.5"
        "test.expected_materials=tungsten")
    set(expected_text "[0,0.5) so a resolved cell has a unique dominant material")
elseif(TEST_CASE STREQUAL "capacity")
    set(test_arguments
        "materials.names=a b c d e f g h i")
    set(expected_text "materials.names supports at most 8 registered materials")
elseif(TEST_CASE STREQUAL "unknown_species")
    set(test_arguments
        "${TEST_INPUT}"
        "materials.foam.species=unknown_foam_carrier")
    set(expected_text "unknown_foam_carrier")
else()
    message(FATAL_ERROR "Unknown TEST_CASE='${TEST_CASE}'")
endif()

execute_process(
    COMMAND "${TEST_EXE}" ${test_arguments}
    RESULT_VARIABLE test_result
    OUTPUT_VARIABLE test_stdout
    ERROR_VARIABLE test_stderr)

set(test_output "${test_stdout}\n${test_stderr}")
if(test_result EQUAL 0)
    message(FATAL_ERROR
        "Expected ${TEST_CASE} to fail, but it returned success.\n${test_output}")
endif()
string(FIND "${test_output}" "${expected_text}" expected_position)
if(expected_position EQUAL -1)
    message(FATAL_ERROR
        "Expected failure text '${expected_text}' was not found.\n${test_output}")
endif()
message(STATUS "Observed expected ${TEST_CASE} rejection: ${expected_text}")
