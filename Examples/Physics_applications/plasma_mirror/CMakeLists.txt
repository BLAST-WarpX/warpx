# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_plasma_mirror  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_plasma_mirror  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)
