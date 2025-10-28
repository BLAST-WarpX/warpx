# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_bilinear_filter  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_bilinear_filter  # inputs
    "analysis.py diags/diag1000001"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_synchronize_velocity  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_synchronize_velocity  # inputs
    "analysis_synchronize_velocity.py diags/diag1000005"  # analysis
    OFF  # checksum
    OFF  # dependency
)
