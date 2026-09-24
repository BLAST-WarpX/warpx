# Each test runs its cases and their analysis together.
if(NOT WarpX_APP OR NOT WarpX_MPI OR NOT "2" IN_LIST WarpX_DIMS)
    return()
endif()

if(NOT Python_EXECUTABLE)
    find_package(Python COMPONENTS Interpreter REQUIRED)
endif()

add_warpx_test(
    test_2d_load_balance_split_merge  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_load_balance_split_merge  # inputs
    OFF  # analysis: included in the run wrapper
    OFF  # checksum: disabled; the run wrapper checks the results
    OFF  # dependency
    RUN_WRAPPER ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_test.py split_merge
)

add_warpx_test(
    test_2d_load_balance_costs  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_load_balance_costs  # inputs
    OFF  # analysis: included in the run wrapper
    OFF  # checksum: disabled; the run wrapper checks the results
    OFF  # dependency
    RUN_WRAPPER ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_test.py costs
)
