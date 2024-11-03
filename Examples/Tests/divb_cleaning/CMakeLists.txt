# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_divb_cleaning  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_divb_cleaning  # inputs
    analysis.py  # analysis
    diags/diag1000400  # output
    OFF  # dependency
)
