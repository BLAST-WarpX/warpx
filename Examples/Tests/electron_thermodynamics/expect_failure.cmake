# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

set(expected_text
    "hybrid_pic_model.electron_thermodynamics=singularity_spiner requires WarpX_PRECISION=DOUBLE; SINGLE field precision is not qualified for tabulated electron-energy source and ledger updates.")

execute_process(
    COMMAND "${TEST_EXE}"
        "hybrid_pic_model.electron_thermodynamics=singularity_spiner"
        "amrex.abort_on_unused_inputs=1"
        "amrex.throw_exception=1"
        "amrex.signal_handling=0"
    RESULT_VARIABLE test_result
    OUTPUT_VARIABLE test_stdout
    ERROR_VARIABLE test_stderr)

set(test_output "${test_stdout}\n${test_stderr}")
if(test_result EQUAL 0)
    message(FATAL_ERROR
        "Expected SINGLE-precision Singularity-Spiner selection to fail, but "
        "it returned success.\n${test_output}")
endif()
string(FIND "${test_output}" "${expected_text}" expected_position)
if(expected_position EQUAL -1)
    message(FATAL_ERROR
        "Expected failure text '${expected_text}' was not found.\n${test_output}")
endif()
message(STATUS "Observed expected SINGLE-precision rejection: ${expected_text}")
