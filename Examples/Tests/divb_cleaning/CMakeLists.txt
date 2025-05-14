# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_divb_cleaning  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_divb_cleaning  # inputs
    "analysis.py diags/diag1000400"  # analysis
    "analysis_default_regression.py --path diags/diag1000400"  # checksum
    OFF  # dependency
)
