# Copyright 2026 The WarpX Community
#
# License: BSD-3-Clause-LBNL

execute_process(
    COMMAND "${TEST_EXE}" "${TEST_INPUT}"
        "amrex.abort_on_unused_inputs=1"
        "amrex.throw_exception=1"
        "amrex.signal_handling=0"
        "warpx.always_warn_immediately=1"
        "warpx.do_dynamic_scheduling=0"
        "warpx.serialize_initial_conditions=1"
    RESULT_VARIABLE test_result
    OUTPUT_VARIABLE test_stdout
    ERROR_VARIABLE test_stderr)

set(test_output "${test_stdout}\n${test_stderr}")
if(test_result EQUAL 0)
    message(FATAL_ERROR
        "Expected invalid hybrid-ionization gamma to fail, but it returned "
        "success.\n${test_output}")
endif()
set(expected_text
    "do_hybrid_ionization requires a finite hybrid_pic_model.gamma > 1")
string(FIND "${test_output}" "${expected_text}" expected_position)
if(expected_position EQUAL -1)
    message(FATAL_ERROR
        "Expected failure text '${expected_text}' was not found.\n${test_output}")
endif()
message(STATUS "Observed expected invalid-gamma rejection: ${expected_text}")
