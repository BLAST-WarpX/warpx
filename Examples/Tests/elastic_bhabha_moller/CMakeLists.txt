# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_bhabha  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_bhabha  # inputs
    analysis_bhabha.py  # analysis
    "analysis_default_regression.py --path diags/diag1"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_moller  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_moller  # inputs
    "analysis_moller.py"  # analysis
    "analysis_default_regression.py --path diags/diag1"  # checksum
    OFF  # dependency
)
