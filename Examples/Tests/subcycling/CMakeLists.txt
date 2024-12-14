# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_subcycling_mr  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_subcycling_mr  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000250"  # checksum
    OFF  # dependency
)
