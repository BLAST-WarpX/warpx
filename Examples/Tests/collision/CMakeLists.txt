# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_collision_z  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_collision_z  # inputs
    analysis_collision_1d.py  # analysis
    diags/diag1000600  # output
    OFF  # dependency
)

add_warpx_test(
    test_2d_collision_xz  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_collision_xz  # inputs
    analysis_collision_2d.py  # analysis
    diags/diag1000150  # output
    OFF  # dependency
)

add_warpx_test(
    test_2d_collision_xz_picmi  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_collision_xz_picmi.py  # inputs
    analysis_collision_2d.py  # analysis
    diags/diag1000150  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_collision_iso  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_collision_iso  # inputs
    analysis_collision_3d_isotropization.py  # analysis
    diags/diag1000100  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_collision_xyz  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_collision_xyz  # inputs
    analysis_collision_3d.py  # analysis
    diags/diag1000150  # output
    OFF  # dependency
)

add_warpx_test(
    test_rz_collision  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_collision  # inputs
    analysis_collision_rz.py  # analysis
    diags/diag1000150  # output
    OFF  # dependency
)
