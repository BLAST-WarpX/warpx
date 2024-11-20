# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_relativistic_space_charge_initialization  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_relativistic_space_charge_initialization  # inputs
    analysis.py  # analysis
    diags/diag1000001  # output
    OFF  # dependency
)
