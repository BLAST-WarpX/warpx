# Configuration ###############################################################
#
if(WarpX_MPI)
    # OpenMPI root guard: https://github.com/open-mpi/ompi/issues/4451
    if("$ENV{USER}" STREQUAL "root")
        # calling even --help as root will abort and warn on stderr
        execute_process(
            COMMAND ${MPIEXEC_EXECUTABLE} --help
            ERROR_VARIABLE MPIEXEC_HELP_TEXT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(${MPIEXEC_HELP_TEXT} MATCHES "^.*allow-run-as-root.*$")
            set(MPI_ALLOW_ROOT --allow-run-as-root)
        endif()
    endif()
endif()

# Add a WarpX test set (with sub-tests)
#
# name: unique name of this test
# dims: 1,2,RZ,3
# nprocs: 1 or 2 (maybe refactor later on to just depend on WarpX_MPI)
# inputs: inputs file or PICMI script, WarpX_MPI decides w/ or w/o MPI
# analysis: custom test analysis command, always run without MPI
# checksum: default regression analysis command (checksum benchmark)
# dependency: name of base test that must run first
#
function(add_warpx_test
    name
    dims
    nprocs
    inputs
    analysis
    checksum
    dependency
)
    # cannot run MPI tests w/o MPI build
    if(nprocs GREATER_EQUAL 2 AND NOT WarpX_MPI)
        message(WARNING "${name}: cannot run MPI tests without MPI build")
        return()
    endif()

    # cannot run tests with unsupported geometry
    if(NOT dims IN_LIST WarpX_DIMS)
        return()
    endif()

    # cannot run tests with unfulfilled dependencies
    if(dependency AND NOT TEST ${dependency}.run)
        return()
    endif()

    # set dimension suffix
    warpx_set_suffix_dims(SD ${dims})

    # make a unique run directory
    file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name})
    set(THIS_WORKING_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name})

    # get input file/script and optional command-line arguments
    separate_arguments(INPUTS_LIST UNIX_COMMAND "${inputs}")
    list(GET INPUTS_LIST 0 INPUTS_FILE)
    list(LENGTH INPUTS_LIST INPUTS_LIST_LENGTH)
    if(INPUTS_LIST_LENGTH GREATER 1)
        list(SUBLIST INPUTS_LIST 1 -1 INPUTS_ARGS)
        list(JOIN INPUTS_ARGS " " INPUTS_ARGS)
    else()
        set(INPUTS_ARGS "")
    endif()

    # get analysis script and optional command-line arguments
    separate_arguments(ANALYSIS_LIST UNIX_COMMAND "${analysis}")
    list(GET ANALYSIS_LIST 0 ANALYSIS_FILE)
    cmake_path(SET ANALYSIS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${ANALYSIS_FILE}")
    list(LENGTH ANALYSIS_LIST ANALYSIS_LIST_LENGTH)
    if(ANALYSIS_LIST_LENGTH GREATER 1)
        list(SUBLIST ANALYSIS_LIST 1 -1 ANALYSIS_ARGS)
        list(JOIN ANALYSIS_ARGS " " ANALYSIS_ARGS)
    else()
        set(ANALYSIS_ARGS "")
    endif()

    # get checksum script and optional command-line arguments
    separate_arguments(CHECKSUM_LIST UNIX_COMMAND "${checksum}")
    list(GET CHECKSUM_LIST 0 CHECKSUM_FILE)
    cmake_path(SET CHECKSUM_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${CHECKSUM_FILE}")
    list(LENGTH CHECKSUM_LIST CHECKSUM_LIST_LENGTH)
    if(CHECKSUM_LIST_LENGTH GREATER 1)
        list(SUBLIST CHECKSUM_LIST 1 -1 CHECKSUM_ARGS)
        list(JOIN CHECKSUM_ARGS " " CHECKSUM_ARGS)
    else()
        set(CHECKSUM_ARGS "")
    endif()

    # Python test?
    set(python OFF)
    if(${INPUTS_FILE} MATCHES ".*\.py$")
        set(python ON)
        cmake_path(SET INPUTS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${INPUTS_FILE}")
    endif()

    # cannot run Python tests w/o Python support
    if(python AND NOT WarpX_PYTHON)
        return()
    endif()

    # cannot run executable tests w/o WarpX executable application
    if(NOT python AND NOT WarpX_APP)
        return()
    endif()

    # set MPI executable
    set(THIS_MPI_TEST_EXE
        ${MPIEXEC_EXECUTABLE}
        ${MPI_ALLOW_ROOT}
        ${MPIEXEC_NUMPROC_FLAG} ${nprocs}
        ${MPIEXEC_POSTFLAGS}
        ${MPIEXEC_PREFLAGS}
    )

    # set Python executable
    if(python)
        set(THIS_Python_EXE ${Python_EXECUTABLE})
    else()
        set(THIS_Python_EXE "")
    endif()

    # test run
    if(python)
        # for argparse, do not pass command-line arguments as one quoted string
        separate_arguments(INPUTS_ARGS UNIX_COMMAND "${INPUTS_ARGS}")
        add_test(
            NAME ${name}.run
            COMMAND
                ${THIS_MPI_TEST_EXE}
                ${THIS_Python_EXE}
                ${INPUTS_FILE}
                ${INPUTS_ARGS}
            WORKING_DIRECTORY ${THIS_WORKING_DIR}
        )
        # FIXME Use helper function to handle Windows exceptions
        set_property(TEST ${name}.run APPEND PROPERTY ENVIRONMENT "PYTHONPATH=$ENV{PYTHONPATH}:${CMAKE_PYTHON_OUTPUT_DIRECTORY}")
    else()
        # TODO Use these for Python tests too
        set(runtime_params
            "amrex.abort_on_unused_inputs = 1"
            "amrex.throw_exception = 1"
            "warpx.always_warn_immediately = 1"
            "warpx.do_dynamic_scheduling = 0"
            "warpx.serialize_initial_conditions = 1"
            # FIXME should go before input file
            #"warpx.abort_on_warning_threshold = low"
        )
        set(runtime_params_fpetrap "")
        if(WarpX_TEST_FPETRAP)
            set(runtime_params_fpetrap
                "amrex.fpe_trap_invalid = 1"
                "amrex.fpe_trap_overflow = 1"
                "amrex.fpe_trap_zero = 1"
            )
        endif()
        add_test(
            NAME ${name}.run
            COMMAND
                ${THIS_MPI_TEST_EXE}
                $<TARGET_FILE:app_${SD}>
                ${INPUTS_FILE}
                ${runtime_params}
                ${runtime_params_fpetrap}
                ${INPUTS_ARGS}
            WORKING_DIRECTORY ${THIS_WORKING_DIR}
        )
    endif()

    # AMReX ParmParse prefix: FILE = <prefix><filename>
    set_property(TEST ${name}.run APPEND PROPERTY ENVIRONMENT "AMREX_INPUTS_FILE_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/")

    # run all tests with 1 OpenMP thread by default
    set_property(TEST ${name}.run APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    if(python OR WIN32)
        set(THIS_Python_SCRIPT_EXE ${Python_EXECUTABLE})
    else()
        set(THIS_Python_SCRIPT_EXE "")
    endif()

    # test analysis
    if(analysis)
        # for argparse, do not pass command-line arguments as one quoted string
        separate_arguments(ANALYSIS_ARGS UNIX_COMMAND "${ANALYSIS_ARGS}")
        add_test(
            NAME ${name}.analysis
            COMMAND
                ${THIS_Python_SCRIPT_EXE}
                ${ANALYSIS_FILE}
                ${ANALYSIS_ARGS}
            WORKING_DIRECTORY ${THIS_WORKING_DIR}
        )
        # test analysis depends on test run
        set_property(TEST ${name}.analysis APPEND PROPERTY DEPENDS "${name}.run")
        # FIXME Use helper function to handle Windows exceptions
        set(PYTHONPATH "$ENV{PYTHONPATH}:${CMAKE_PYTHON_OUTPUT_DIRECTORY}")
        # add paths for custom Python modules
        set(PYTHONPATH "${PYTHONPATH}:${WarpX_SOURCE_DIR}/Regression/PostProcessingUtils")
        set(PYTHONPATH "${PYTHONPATH}:${WarpX_SOURCE_DIR}/Tools/Parser")
        set(PYTHONPATH "${PYTHONPATH}:${WarpX_SOURCE_DIR}/Tools/PostProcessing")
        set_property(TEST ${name}.analysis APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${PYTHONPATH}")
    endif()

    # checksum analysis
    if(checksum)
        # for argparse, do not pass command-line arguments as one quoted string
        separate_arguments(CHECKSUM_ARGS UNIX_COMMAND "${CHECKSUM_ARGS}")
        add_test(
            NAME ${name}.checksum
            COMMAND
                ${THIS_Python_SCRIPT_EXE}
                ${CHECKSUM_FILE}
                ${CHECKSUM_ARGS}
            WORKING_DIRECTORY ${THIS_WORKING_DIR}
        )
        # test analysis depends on test run
        set_property(TEST ${name}.checksum APPEND PROPERTY DEPENDS "${name}.run")
        if(analysis)
            # checksum analysis depends on test analysis
            set_property(TEST ${name}.checksum APPEND PROPERTY DEPENDS "${name}.analysis")
        endif()
        # FIXME Use helper function to handle Windows exceptions
        set(PYTHONPATH "$ENV{PYTHONPATH}:${CMAKE_PYTHON_OUTPUT_DIRECTORY}")
        # add paths for custom Python modules
        set(PYTHONPATH "${PYTHONPATH}:${WarpX_SOURCE_DIR}/Regression/Checksum")
        set_property(TEST ${name}.checksum APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${PYTHONPATH}")
    endif()

    # CI: remove test directory after run
    if(WarpX_TEST_CLEANUP)
        add_test(
            NAME ${name}.cleanup
            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/Examples/test_cleanup.cmake ${THIS_WORKING_DIR}
        )
        # test cleanup depends on test run
        set_property(TEST ${name}.cleanup APPEND PROPERTY DEPENDS "${name}.run")
        if(analysis)
            # test cleanup depends on test analysis
            set_property(TEST ${name}.cleanup APPEND PROPERTY DEPENDS "${name}.analysis")
        endif()
        if(checksum)
            # test cleanup depends on test analysis
            set_property(TEST ${name}.cleanup APPEND PROPERTY DEPENDS "${name}.checksum")
        endif()
    endif()

    # Do we depend on another test?
    if(dependency)
        # current test depends on dependency test run (and analysis)
        set_property(TEST ${name}.run APPEND PROPERTY DEPENDS "${dependency}.run")
        if(analysis)
            set_property(TEST ${name}.run APPEND PROPERTY DEPENDS "${dependency}.analysis")
        endif()
        if(checksum)
            set_property(TEST ${name}.run APPEND PROPERTY DEPENDS "${dependency}.checksum")
        endif()
        if(WarpX_TEST_CLEANUP)
            # do not clean up dependency test before current test is completed
            set_property(TEST ${dependency}.cleanup APPEND PROPERTY DEPENDS "${name}.cleanup")
        endif()
    endif()
endfunction()

# Add a CTest label to a WarpX test set.
#
# Labeling it here will add the label to the run test, its analysis and its cleanup.
#
# name: unique name of this test
# label: ctest LABELS property value to be added
#
function(label_warpx_test name label)
    set(_test_names "${name}.run;${name}.analysis;${name}.cleanup")
    foreach(_test_name IN LISTS _test_names)
        if(TEST ${_test_name})
            set_property(TEST ${_test_name} APPEND PROPERTY LABELS "${label}")
        endif()
    endforeach()
endfunction()

# Add tests (alphabetical order) ##############################################
#

add_subdirectory(Tests)
add_subdirectory(Physics_applications)
