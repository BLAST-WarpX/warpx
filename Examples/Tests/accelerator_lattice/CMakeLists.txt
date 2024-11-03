# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_hard_edged_quadrupoles  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_hard_edged_quadrupoles  # inputs
    analysis.py  # analysis
    diags/diag1000050  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_hard_edged_quadrupoles_boosted  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_hard_edged_quadrupoles_boosted  # inputs
    analysis.py  # analysis
    diags/diag1000050  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_hard_edged_quadrupoles_moving  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_hard_edged_quadrupoles_moving  # inputs
    analysis.py  # analysis
    diags/diag1000050  # output
    OFF  # dependency
)
