# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_collider_diagnostics  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_collider_diagnostics  # inputs
    analysis.py  # analysis
    diags/diag1000001  # output
    OFF  # dependency
)
