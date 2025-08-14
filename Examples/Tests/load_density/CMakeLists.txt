# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_load_density_prepare  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_load_density_prepare.py  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_load_density  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_load_density  # inputs
    "analysis_3d.py"  # analysis
    "analysis_default_regression.py --path diags/diag/"   # checksum
    test_3d_load_density_prepare  # dependency
)
