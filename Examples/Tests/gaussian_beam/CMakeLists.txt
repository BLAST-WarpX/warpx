# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_focusing_gaussian_beam  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_focusing_gaussian_beam  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000000"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_gaussian_beam_picmi  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_gaussian_beam_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)
