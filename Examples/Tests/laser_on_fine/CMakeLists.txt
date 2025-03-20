# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_laser_on_fine  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_laser_on_fine  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000050"  # checksum
    OFF  # dependency
)
