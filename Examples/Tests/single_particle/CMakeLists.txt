# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_bilinear_filter  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_bilinear_filter  # inputs
    analysis.py  # analysis
    diags/diag1000001  # output
    OFF  # dependency
)
