# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_initial_distribution  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_initial_distribution  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)
