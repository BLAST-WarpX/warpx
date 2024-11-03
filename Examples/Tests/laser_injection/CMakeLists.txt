# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_laser_injection  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_laser_injection  # inputs
    analysis_1d.py  # analysis
    diags/diag1000240  # output
    OFF  # dependency
)

add_warpx_test(
    test_2d_laser_injection  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_laser_injection  # inputs
    analysis_2d.py  # analysis
    diags/diag1000240  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_laser_injection  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_laser_injection  # inputs
    analysis_3d.py  # analysis
    diags/diag1000020  # output
    OFF  # dependency
)
