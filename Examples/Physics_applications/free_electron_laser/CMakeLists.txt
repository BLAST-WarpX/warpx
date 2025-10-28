# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_fel  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_fel  # inputs
    "analysis_fel.py diags/diag_labframe"  # analysis
    "analysis_default_regression.py --path diags/diag_labframe"  # checksum
    OFF  # dependency
)
