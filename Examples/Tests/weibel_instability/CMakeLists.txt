# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_weibel_instability_darwin  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_weibel_instability_darwin  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1/"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_weibel_instability_explicit  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_weibel_instability_explicit  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1/"  # checksum
    OFF  # dependency
)
