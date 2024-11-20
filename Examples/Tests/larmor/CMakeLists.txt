# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_larmor  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_larmor  # inputs
    analysis_default_regression.py  # analysis
    diags/diag1000010  # output
    OFF  # dependency
)
