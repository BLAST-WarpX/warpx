# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_radiation_reaction  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_radiation_reaction  # inputs
    "analysis.py diags/diag1000064"  # analysis
    "analysis_default_regression.py --path diags/diag1000064"  # checksum
    OFF  # dependency
)
