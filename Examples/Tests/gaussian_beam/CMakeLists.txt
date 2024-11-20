# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_focusing_gaussian_beam  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_focusing_gaussian_beam  # inputs
    analysis.py  # analysis
    diags/diag1000000  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_gaussian_beam_picmi  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_gaussian_beam_picmi.py  # inputs
    analysis_default_regression.py  # analysis
    diags/diag1000010  # output
    OFF  # dependency
)
