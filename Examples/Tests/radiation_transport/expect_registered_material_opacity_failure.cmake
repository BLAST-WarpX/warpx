# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

if(TEST_CASE STREQUAL "mixed")
    set(test_input
        "inputs_base_1d_radiation_transport_registered_material_opacity_mixed")
    set(material_b_table "${FIXTURE_DIR}/material_b.h5")
    set(expected_text "radiation or material mutation: 4 owned mixed cell(s)")
elseif(TEST_CASE STREQUAL "group_mismatch")
    set(test_input
        "inputs_base_1d_radiation_transport_registered_material_opacity_order_a")
    set(material_b_table "${FIXTURE_DIR}/material_b_group_mismatch.h5")
    set(expected_text "finite internal edges in Registered opacity table")
elseif(TEST_CASE STREQUAL "key_mismatch")
    set(test_input
        "inputs_base_1d_radiation_transport_registered_material_opacity_order_a")
    set(material_b_table "${FIXTURE_DIR}/material_b_key_mismatch.h5")
    set(expected_text "opacity table metadata does not match named material")
else()
    message(FATAL_ERROR "Unknown TEST_CASE='${TEST_CASE}'")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "AMREX_INPUTS_FILE_PREFIX=${INPUT_PREFIX}/"
        "${TEST_EXE}"
        "${test_input}"
        "materials.material_a.opacity_table_file=${FIXTURE_DIR}/material_a.h5"
        "materials.material_b.opacity_table_file=${material_b_table}"
        "amrex.abort_on_unused_inputs=1"
        "amrex.throw_exception=1"
        "amrex.the_arena_init_size=0"
        "warpx.serialize_initial_conditions=1"
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
