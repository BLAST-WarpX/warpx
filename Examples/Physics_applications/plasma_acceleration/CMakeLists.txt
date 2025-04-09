# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_plasma_acceleration_picmi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_plasma_acceleration_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1001000"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_plasma_acceleration_boosted  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_plasma_acceleration_boosted  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_plasma_acceleration_mr  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_plasma_acceleration_mr  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000400"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_plasma_acceleration_mr_momentum_conserving  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_plasma_acceleration_mr_momentum_conserving  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000400"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_plasma_acceleration_boosted  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_plasma_acceleration_boosted  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000005"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_plasma_acceleration_boosted_hybrid  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_plasma_acceleration_boosted_hybrid  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000025"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_plasma_acceleration_mr_picmi  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_plasma_acceleration_mr_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000002"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_plasma_acceleration_picmi  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_plasma_acceleration_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)
