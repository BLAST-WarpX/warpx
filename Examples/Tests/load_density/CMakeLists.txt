# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_load_density_prepare  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_load_density_prepare.py  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_load_density  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_load_density  # inputs
    "analysis_1d.py"  # analysis
    "analysis_default_regression.py --path diags/diag/"   # checksum
    test_1d_load_density_prepare  # dependency
)

add_warpx_test(
    test_2d_load_density_prepare  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_load_density_prepare.py  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_load_density  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_load_density  # inputs
    "analysis_2d.py"  # analysis
    "analysis_default_regression.py --path diags/diag/"   # checksum
    test_2d_load_density_prepare  # dependency
)

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

add_warpx_test(
    test_rz_load_density_prepare  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_load_density_prepare.py  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_load_density  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_load_density  # inputs
    "analysis_rz.py"  # analysis
    "analysis_default_regression.py --path diags/diag/"   # checksum
    test_rz_load_density_prepare  # dependency
)
