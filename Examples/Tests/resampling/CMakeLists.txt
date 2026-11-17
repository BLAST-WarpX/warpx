# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_resample_velocity_coincidence_thinning  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_resample_velocity_coincidence_thinning  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000004"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_resample_velocity_coincidence_thinning_cartesian  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_resample_velocity_coincidence_thinning_cartesian  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000004"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_leveling_thinning  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_leveling_thinning  # inputs
    "analysis.py diags/diag1000008"  # analysis
    "analysis_default_regression.py --path diags/diag1000008"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_resample_velocity_coincidence_thinning  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_resample_velocity_coincidence_thinning  # inputs
    "analysis_rz_velocity_coincidence_thinning.py diags/diag1000001"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)
