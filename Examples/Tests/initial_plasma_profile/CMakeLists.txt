# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_parabolic_channel_initialization  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_parabolic_channel_initialization  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000001 --skip-particles --rtol 1e-4"  # checksum
    OFF  # dependency
)
