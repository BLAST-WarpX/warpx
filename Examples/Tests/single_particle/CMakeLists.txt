# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_bilinear_filter  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_bilinear_filter  # inputs
    "analysis.py diags/diag1000001"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)
