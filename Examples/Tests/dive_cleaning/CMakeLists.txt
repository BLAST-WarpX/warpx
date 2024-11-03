# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_dive_cleaning  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_dive_cleaning  # inputs
    analysis.py  # analysis
    diags/diag1000128  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_dive_cleaning  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_dive_cleaning  # inputs
    analysis.py  # analysis
    diags/diag1000128  # output
    OFF  # dependency
)
