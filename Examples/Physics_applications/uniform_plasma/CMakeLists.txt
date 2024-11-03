# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_uniform_plasma  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_uniform_plasma  # inputs
    analysis_default_regression.py  # analysis
    diags/diag1000010  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_uniform_plasma  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_uniform_plasma  # inputs
    analysis_default_regression.py  # analysis
    diags/diag1000010  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_uniform_plasma_restart  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_uniform_plasma_restart  # inputs
    analysis_default_restart.py  # analysis
    diags/diag1000010  # output
    test_3d_uniform_plasma  # dependency
)
