# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_dive_cleaning  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_dive_cleaning  # inputs
    "analysis.py diags/diag1000128"  # analysis
    "analysis_default_regression.py --path diags/diag1000128"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_dive_cleaning  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_dive_cleaning  # inputs
    "analysis.py diags/diag1000128"  # analysis
    "analysis_default_regression.py --path diags/diag1000128"  # checksum
    OFF  # dependency
)
