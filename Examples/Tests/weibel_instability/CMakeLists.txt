# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_weibel_instability  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_weibel_instability  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1/"  # checksum
    OFF  # dependency
)
