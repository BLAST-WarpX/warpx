# CTest's PASS_REGULAR_EXPRESSION does not require a failed child process. Use
# the shared command-contract wrapper so unsupported configurations must both
# fail and emit their intended diagnostic before accepting a timestep. An
# optional fifth argument requests an MPI rank count; the default is one.
function(add_radiation_failure_test name dims inputs expected)
    set(nprocs 1)
    if(ARGC GREATER 4)
        set(nprocs ${ARGV4})
    endif()
    if(nprocs GREATER_EQUAL 2 AND NOT WarpX_MPI)
        return()
    endif()
    if(NOT ${dims} IN_LIST WarpX_DIMS OR NOT WarpX_APP)
        return()
    endif()
    if(NOT Python_EXECUTABLE)
        find_package(Python COMPONENTS Interpreter REQUIRED)
    endif()
    separate_arguments(input_args UNIX_COMMAND "${inputs}")
    warpx_set_suffix_dims(SD ${dims})
    set(test_working_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}")
    file(MAKE_DIRECTORY "${test_working_dir}")
    set(test_mpi_command "")
    if(nprocs GREATER_EQUAL 2)
        set(test_mpi_command
            ${MPIEXEC_EXECUTABLE}
            ${MPI_ALLOW_ROOT}
            ${MPIEXEC_NUMPROC_FLAG} ${nprocs}
            ${MPIEXEC_POSTFLAGS}
            ${MPIEXEC_PREFLAGS})
    endif()
    add_test(
        NAME ${name}.run
        COMMAND
            ${Python_EXECUTABLE}
            ${WarpX_SOURCE_DIR}/Examples/Tests/implicit/check_command_output.py
            --expect-failure
            --expect "${expected}"
            --forbid "STEP 1 ends"
            --
            ${test_mpi_command}
            $<TARGET_FILE:app_${SD}>
            ${input_args}
            "amrex.abort_on_unused_inputs = 1"
            "amrex.throw_exception = 1"
            "amrex.the_arena_init_size = 0"
            "warpx.always_warn_immediately = 1"
            "warpx.do_dynamic_scheduling = 0"
            "warpx.serialize_initial_conditions = 1"
        WORKING_DIRECTORY "${test_working_dir}")
    set_property(
        TEST ${name}.run APPEND PROPERTY
        ENVIRONMENT
            "AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/"
            "OMP_NUM_THREADS=1")
endfunction()
