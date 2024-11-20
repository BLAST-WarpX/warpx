# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_laser_on_fine  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_laser_on_fine  # inputs
    analysis_default_regression.py  # analysis
    diags/diag1000050  # output
    OFF  # dependency
)
