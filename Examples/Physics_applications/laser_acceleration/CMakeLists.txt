# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_laser_acceleration  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_laser_acceleration  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_laser_acceleration_fluid_boosted  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_laser_acceleration_fluid_boosted  # inputs
    "analysis_1d_fluid_boosted.py diags/diag1000001"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)
label_warpx_test(test_1d_laser_acceleration_fluid_boosted slow)

add_warpx_test(
    test_1d_laser_acceleration_picmi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_laser_acceleration_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_laser_acceleration_boosted  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_laser_acceleration_boosted  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000002"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_laser_acceleration_mr  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_laser_acceleration_mr  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000200"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_laser_acceleration_mr_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_laser_acceleration_mr_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000200"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_refined_injection  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_refined_injection  # inputs
    "analysis_refined_injection.py diags/diag1000200"  # analysis
    "analysis_default_regression.py --path diags/diag1000200"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_laser_acceleration  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_laser_acceleration  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1/"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_laser_acceleration_picmi  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_laser_acceleration_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_laser_acceleration_single_precision_comms  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_laser_acceleration_single_precision_comms  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1/"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_laser_acceleration  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_laser_acceleration  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_laser_acceleration_opmd  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_laser_acceleration_opmd  # inputs
    "analysis_openpmd_rz.py diags/diag1/"  # analysis
    "analysis_default_regression.py --path diags/diag1/"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_laser_acceleration_picmi  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_laser_acceleration_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)
label_warpx_test(test_rz_laser_acceleration_picmi slow)
