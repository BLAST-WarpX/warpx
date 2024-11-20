# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_rigid_injection_btd  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_rigid_injection_btd  # inputs
    analysis_rigid_injection_btd.py  # analysis
    diags/diag1000001  # output
    OFF  # dependency
)

add_warpx_test(
    test_2d_rigid_injection_lab  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_rigid_injection_lab  # inputs
    analysis_rigid_injection_lab.py  # analysis
    diags/diag1000289  # output
    OFF  # dependency
)
