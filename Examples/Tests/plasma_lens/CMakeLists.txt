# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_plasma_lens  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_plasma_lens  # inputs
    analysis.py  # analysis
    diags/diag1000084  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_plasma_lens_boosted  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_plasma_lens_boosted  # inputs
    analysis.py  # analysis
    diags/diag1000084  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_plasma_lens_hard_edged  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_plasma_lens_hard_edged  # inputs
    analysis.py  # analysis
    diags/diag1000084  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_plasma_lens_picmi  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_plasma_lens_picmi.py  # inputs
    analysis.py  # analysis
    diags/diag1000084  # output
    OFF  # dependency
)

add_warpx_test(
    test_3d_plasma_lens_short  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_plasma_lens_short  # inputs
    analysis.py  # analysis
    diags/diag1000084  # output
    OFF  # dependency
)
