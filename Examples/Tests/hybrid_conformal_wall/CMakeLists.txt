# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_hybrid_conformal_wall  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_hybrid_conformal_wall  # inputs
    "analysis_wall.py diags/diag1000003"  # analysis
    "analysis_default_regression.py --path diags/diag1000003"  # checksum
    OFF  # dependency
)
