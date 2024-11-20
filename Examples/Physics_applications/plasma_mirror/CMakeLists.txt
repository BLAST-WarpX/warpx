# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_plasma_mirror  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_plasma_mirror  # inputs
    analysis_default_regression.py  # analysis
    diags/diag1000020  # output
    OFF  # dependency
)
