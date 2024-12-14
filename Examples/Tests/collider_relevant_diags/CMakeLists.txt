# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_collider_diagnostics  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_collider_diagnostics  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)
