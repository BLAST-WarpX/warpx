# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_rigid_injection_btd  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_rigid_injection_btd  # inputs
    "analysis_rigid_injection_btd.py diags/diag1000001"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_rigid_injection_lab  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_rigid_injection_lab  # inputs
    "analysis_rigid_injection_lab.py diags/diag1000289"  # analysis
    "analysis_default_regression.py --path diags/diag1000289"  # checksum
    OFF  # dependency
)
