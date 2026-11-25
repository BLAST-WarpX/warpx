# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_laser_injection_from_lasy_file  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_laser_injection_from_lasy_file  # inputs
    "analysis_1d.py diags/diag1000251"  # analysis
    "analysis_default_regression.py --path diags/diag1000251"  # checksum
    OFF  # dependency
    PREPARE test_1d_laser_injection_from_lasy_file_prepare.py
)

add_warpx_test(
    test_1d_laser_injection_from_lasy_file_boost  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_laser_injection_from_lasy_file_boost  # inputs
    "analysis_1d_boost.py diags/diag1000001"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
    PREPARE test_1d_laser_injection_from_lasy_file_boost_prepare.py
)

add_warpx_test(
    test_2d_laser_injection_from_binary_file  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_laser_injection_from_binary_file  # inputs
    "analysis_2d_binary.py diags/diag1000250"  # analysis
    "analysis_default_regression.py --path diags/diag1000250"  # checksum
    OFF  # dependency
    PREPARE test_2d_laser_injection_from_binary_file_prepare.py
)

add_warpx_test(
    test_2d_laser_injection_from_lasy_file  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_laser_injection_from_lasy_file  # inputs
    "analysis_2d.py diags/diag1000251"  # analysis
    "analysis_default_regression.py --path diags/diag1000251"  # checksum
    OFF  # dependency
    PREPARE test_2d_laser_injection_from_lasy_file_prepare.py
)

add_warpx_test(
    test_3d_laser_injection_from_lasy_file  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_laser_injection_from_lasy_file  # inputs
    "analysis_3d.py diags/diag1000251"  # analysis
    "analysis_default_regression.py --path diags/diag1000251"  # checksum
    OFF  # dependency
    PREPARE test_3d_laser_injection_from_lasy_file_prepare.py
)

add_warpx_test(
    test_rz_laser_injection_from_lasy_file  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_laser_injection_from_lasy_file  # inputs
    "analysis_rz.py diags/diag1000252"  # analysis
    "analysis_default_regression.py --path diags/diag1000252"  # checksum
    OFF  # dependency
    PREPARE test_rz_laser_injection_from_lasy_file_prepare.py
)

add_warpx_test(
    test_rz_laser_injection_from_RZ_lasy_file  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_laser_injection_from_RZ_lasy_file  # inputs
    "analysis_from_RZ_file.py diags/diag1000612"  # analysis
    "analysis_default_regression.py --path diags/diag1000612"  # checksum
    OFF  # dependency
    PREPARE test_rz_laser_injection_from_RZ_lasy_file_prepare.py
)
