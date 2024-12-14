# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_laser_injection  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_laser_injection  # inputs
    "analysis_1d.py diags/diag1000240"  # analysis
    "analysis_default_regression.py --path diags/diag1000240"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_laser_injection  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_laser_injection  # inputs
    "analysis_2d.py diags/diag1000240"  # analysis
    "analysis_default_regression.py --path diags/diag1000240"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_laser_injection  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_laser_injection  # inputs
    "analysis_3d.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_laser_injection_implicit  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_laser_injection_implicit  # inputs
    "analysis_1d.py diags/diag1000240"  # analysis
    "analysis_default_regression.py --path diags/diag1000240"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_laser_injection_implicit  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_laser_injection_implicit  # inputs
    "analysis_2d.py diags/diag1000240"  # analysis
    "analysis_default_regression.py --path diags/diag1000240"  # checksum
    OFF  # dependency
)
